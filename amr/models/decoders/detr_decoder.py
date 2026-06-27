# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)

import math
import copy
import os
from typing import Optional, List
from utils.misc import inverse_sigmoid

import torch
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch import nn, Tensor
from torch.nn.init import constant_
from xformers.ops import memory_efficient_attention, fmha


def position_encoding_xy(pos_x, pos_y, embedding_dim, temperature = 20, scale = 2*math.pi):
    assert embedding_dim % 2 == 0
    assert pos_x.ndim == 1 and pos_y.ndim == 1
    dim_t = torch.arange(embedding_dim // 2, dtype=torch.float32, device=pos_x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / (embedding_dim // 2))
    x_embed = pos_x * scale
    y_embed = pos_y * scale
    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)
    pos = torch.cat([pos_y,pos_x], dim=1)
    return pos


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_queries=300, 
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.0,
                 activation="relu", 
                 return_intermediate_dec=False, query_dim=4,
                 keep_query_pos=False, query_scale_type='cond_elewise',
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=True,
                 ):

        super().__init__()
        decoder_layer = XformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, keep_query_pos=keep_query_pos)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = XformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=d_model, query_dim=query_dim, keep_query_pos=keep_query_pos, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def mask2bias(self, mask, batch_size):
        if mask is None:
            return None

        assert mask.dtype == torch.bool
        assert mask.ndim == 2
        L, S = mask.shape[0], mask.shape[1]
        pad_size = (S + 7) // 8 * 8
        bias = torch.zeros((batch_size, self.nhead, L, pad_size), device = mask.device)[:,:,:,:S]
        bias.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        return bias


    def forward(self, memory, memory_lens, tgt, tgt_lens, refpoint_embed, pos_embed, self_attn_mask):
        self_attn_bias = self.mask2bias(self_attn_mask, batch_size=len(memory_lens))
        hs, references = self.decoder(memory=memory, memory_lens=memory_lens,
                          tgt=tgt, tgt_lens=tgt_lens,
                          pos=pos_embed, refpoints_unsigmoid=refpoint_embed,
                          self_attn_bias = self_attn_bias)
        return hs, references


class XformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=True, 
                    d_model=512, query_dim=4, keep_query_pos=False, query_scale_type='cond_elewise',
                    modulate_hw_attn=False,
                    bbox_embed_diff_each_layer=False,
                    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, d_model)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        
        self.bbox_embed = None
        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer


        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)

        
        if not keep_query_pos:
            for layer_id in range(num_layers - 1):
                self.layers[layer_id + 1].ca_qpos_proj = None

    def forward(self, memory, memory_lens, tgt, tgt_lens,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None, # L_tgt, 4
                self_attn_bias = None):
        B, num_queries = len(tgt_lens), tgt_lens[0]
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points.view(B, num_queries, self.query_dim)]

        # import ipdb; ipdb.set_trace()        

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[:, :self.query_dim]  # [L_tgt, 4]
            # get sine embedding for the query vector
            xy_embed = position_encoding_xy(obj_center[:,0], obj_center[:,1], self.d_model)
            wh_embed = position_encoding_xy(obj_center[:,2], obj_center[:,3], self.d_model)
            query_sine_embed = torch.cat([xy_embed,wh_embed],dim=1) #[L_tgt, 2*d_model]
            query_pos = self.ref_point_head(query_sine_embed) 

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)
            else:
                pos_transformation = self.query_scale.weight[layer_id]

            # apply transformation
            query_sine_embed = query_sine_embed[:,:self.d_model] * pos_transformation

            # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_center[..., 2]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_center[..., 3]).unsqueeze(-1)


            output = layer(memory=memory, memory_lens=memory_lens,
                           tgt=output, tgt_lens=tgt_lens,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           is_first=(layer_id == 0),
                           self_attn_bias = self_attn_bias)

            # iter update
            if self.bbox_embed is not None:
                if self.bbox_embed_diff_each_layer:
                    tmp = self.bbox_embed[layer_id](self.norm(output))
                else:
                    tmp = self.bbox_embed(self.norm(output))
                # import ipdb; ipdb.set_trace()
                tmp[..., :self.query_dim] += inverse_sigmoid(reference_points)
                new_reference_points = tmp[..., :self.query_dim].sigmoid()
                if layer_id != self.num_layers - 1:
                    ref_points.append(new_reference_points.view(B, num_queries, self.query_dim))
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output).view(B, num_queries, self.d_model))

        # if self.norm is not None:
        #     output = self.norm(output)
        #     if self.return_intermediate:
        #         intermediate.pop()
        #         intermediate.append(output.view(B, num_queries, self.d_model))

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate),
                    torch.stack(ref_points),
                ]
            else:
                return [
                    torch.stack(intermediate), 
                    reference_points.unsqueeze(0)
                ]

        return output.unsqueeze(0)


class XformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.0,
                 activation="relu", keep_query_pos=False):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.sa_out_proj = nn.Linear(d_model, d_model)
        constant_(self.sa_out_proj.bias, 0.)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.ca_out_proj = nn.Linear(d_model, d_model)
        constant_(self.ca_out_proj.bias, 0.)

        self.d_model = d_model
        self.nhead = nhead
        assert self.d_model%self.nhead == 0

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, memory, memory_lens, pos,
                    tgt, tgt_lens, query_pos, query_sine_embed,
                    is_first=False,
                    self_attn_bias=None):
        # self_attn_bias is only used for dn_training
        # 'True' indicates that the element should take part in attention

        B, num_queries = len(tgt_lens), tgt_lens[0]
        L_mem, C_mem = memory.shape
        L_tgt, C_tgt = tgt.shape
        assert C_mem == C_tgt

        # ========== Begin of Self-Attention =============  
        tgt_b4n = tgt
        tgt = self.norm1(tgt)

        q_content = self.sa_qcontent_proj(tgt)
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        q = q_content + q_pos
        k = k_content + k_pos

        q = q.view(B, num_queries, self.nhead, self.d_model // self.nhead)
        k = k.view(B, num_queries, self.nhead, self.d_model // self.nhead)
        v = v.view(B, num_queries, self.nhead, self.d_model // self.nhead)

        tgt2 = memory_efficient_attention(q, k, v, attn_bias=self_attn_bias)
        tgt2 = self.sa_out_proj(tgt2.view(L_tgt, self.d_model))

        tgt = tgt_b4n + self.dropout1(tgt2)
        # ========== End of Self-Attention =============



        # ========== Begin of Cross-Attention =============
        tgt_b4n = tgt
        tgt = self.norm2(tgt)

        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we concatenate the positional embedding predicted from 
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            q_pos = self.ca_qpos_proj(query_pos)
            q = q_content + q_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        q = q.view(1, L_tgt, self.nhead, self.d_model//self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(1, L_tgt, self.nhead, self.d_model//self.nhead)
        q = torch.cat([q, query_sine_embed], dim=3)

        k = k.view(1, L_mem, self.nhead, self.d_model//self.nhead)
        k_pos = k_pos.view(1, L_mem, self.nhead, self.d_model//self.nhead)
        k = torch.cat([k, k_pos], dim=3)

        v = v.view(1, L_mem, self.nhead, self.d_model//self.nhead)

        attn_bias = fmha.attn_bias.BlockDiagonalMask.from_seqlens(q_seqlen = tgt_lens, kv_seqlen = memory_lens)
        tgt2 = memory_efficient_attention(q, k, v, attn_bias=attn_bias)   
        tgt2 = self.ca_out_proj(tgt2.view(L_tgt, self.d_model))             

        tgt = tgt_b4n + self.dropout2(tgt2)
        # ========== End of Cross-Attention =============

        # FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(self.norm3(tgt)))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_decoder(args):
    return TransformerDecoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        num_queries=args.num_queries,
        dim_feedforward=args.dim_feedforward,
        num_decoder_layers=args.dec_layers,
        return_intermediate_dec=True,
        query_dim=4,
        activation=args.transformer_activation
    )


def torch_attention(query, key, value, attn_bias = None):
    scale = 1.0 / query.shape[-1] ** 0.5
    query = query * scale
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    attn = query @ key.transpose(-2, -1)
    if attn_bias is not None:
        attn = attn + attn_bias
    attn = attn.softmax(-1)
    # attn = F.dropout(attn, p)
    attn = attn @ value
    return attn.transpose(1, 2)