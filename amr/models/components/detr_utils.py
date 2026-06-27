import torch


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def prepare_for_cdn(targets, dn_cfg, num_queries, hidden_dim, dn_enc):
    """
        A major difference of DINO from DN-DETR is that the author process pattern embedding pattern embedding in its detector
        forward function and use learnable tgt embedding, so we change this function a little bit.
        :param dn_args: targets, dn_number, label_noise_ratio, box_noise_scale
        :param training: if it is training or inference
        :param num_queries: number of queires
        :param num_classes: number of classes
        :param hidden_dim: transformer hidden dim
        :param label_enc: encode labels in dn
        :return:
        """
    device = targets[0]['boxes'].device

    dn_number = dn_cfg['dn_number']
    box_noise_scale = dn_cfg['box_noise_scale']
    tgt_noise_scale = dn_cfg['tgt_noise_scale']
    known = [(torch.ones_like(t['labels'])) for t in targets]
    batch_size = len(known)
    known_num = [sum(k) for k in known]

    if int(max(known_num)) == 0:
        dn_number = 1
    else:
        if dn_number >= 100:
            dn_number = dn_number // (int(max(known_num) * 2))
        elif dn_number < 1:
            dn_number = 1
    if dn_number == 0:
        dn_number = 1


    unmask_bbox = torch.cat(known)
    
    boxes = torch.cat([t['boxes'] for t in targets])
    assert boxes.ndim == 2
    batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
    known_indice = torch.nonzero(unmask_bbox)
    known_indice = known_indice.view(-1)
    known_indice = known_indice.repeat(2 * dn_number, 1).view(-1)
    known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)

    single_pad = int(max(known_num))
    pad_size = int(single_pad * 2 * dn_number)
    positive_idx = torch.tensor(range(len(boxes))).long().to(device=device).unsqueeze(0).repeat(dn_number, 1)
    positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().to(device=device).unsqueeze(1)
    positive_idx = positive_idx.flatten()
    negative_idx = positive_idx + len(boxes)

    # box queries
    known_bboxs = boxes.repeat(2 * dn_number, 1)
    known_bbox_expand = known_bboxs.clone()
    if box_noise_scale > 0:
        known_bbox_ = torch.zeros_like(known_bboxs)
        known_bbox_[:, :2] = known_bboxs[:, :2] - known_bboxs[:, 2:] / 2
        known_bbox_[:, 2:] = known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

        diff = torch.zeros_like(known_bboxs)
        diff[:, :2] = known_bboxs[:, 2:] / 2
        diff[:, 2:] = known_bboxs[:, 2:] / 2

        rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
        rand_part = torch.rand_like(known_bboxs)
        rand_part[negative_idx] += 1.0
        rand_part *= rand_sign
        known_bbox_ = known_bbox_ + torch.mul(rand_part,
                                                diff).to(device=device) * box_noise_scale
        known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
        known_bbox_expand[:, :2] = (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
        known_bbox_expand[:, 2:] = known_bbox_[:, 2:] - known_bbox_[:, :2]
    input_bbox_embed = inverse_sigmoid(known_bbox_expand)

    # tgt queries
    if dn_cfg['tgt_embed_type'] == 'labels':
        labels = torch.cat([t['labels'] for t in targets])
        known_labels = labels.repeat(2 * dn_number, 1).view(-1)
        known_labels_expaned = known_labels.clone()
        if tgt_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < tgt_noise_scale).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, dn_cfg['dn_labelbook_size'])  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        m = known_labels_expaned.long().to(device=device)
        input_tgt_embed = dn_enc(m)
    elif dn_cfg['tgt_embed_type'] == 'params':
        poses = torch.cat([t['poses'] for t in targets])
        betas = torch.cat([t['betas'] for t in targets])
        params = torch.cat([poses, betas], dim=-1)
        assert params.ndim == 2
        known_params = params.repeat(2 * dn_number, 1)
        known_params_expaned = known_params.clone()
        if tgt_noise_scale > 0:
            rand_sign = torch.randint_like(known_params, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0
            rand_part = torch.rand_like(known_params)
            rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_params_expaned = known_params_expaned + rand_part * tgt_noise_scale
        m = known_params_expaned.to(device=device)
        input_tgt_embed = dn_enc(m)
    

    padding_tgt = torch.zeros((pad_size, hidden_dim), device=device)
    padding_bbox = torch.zeros((pad_size, 4), device=device)

    input_query_tgt = padding_tgt.repeat(batch_size, 1, 1)
    input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

    map_known_indice = torch.tensor([]).to(device=device)
    if len(known_num):
        map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
        map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long()
    if len(known_bid):
        input_query_tgt[(known_bid.long(), map_known_indice)] = input_tgt_embed
        input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed


    # prepare attn_mask
    tgt_size = pad_size + num_queries
    attn_mask = torch.zeros((tgt_size, tgt_size), dtype=bool, device=device)
    # match query cannot see the reconstruct
    attn_mask[pad_size:, :pad_size] = True
    # reconstruct cannot see each other
    for i in range(dn_number):
        if i == 0:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
        if i == dn_number - 1:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
        else:
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
            attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True

    dn_meta = {
        'pad_size': pad_size,
        'num_dn_group': dn_number,
    }

    return input_query_tgt, input_query_bbox, attn_mask, dn_meta


def dn_post_process(pred_poses, pred_betas,
                    pred_boxes, pred_confs,
                    pred_j3ds, pred_j2ds, pred_depths, 
                    pred_verts, pred_transl,
                    dn_meta, aux_loss, _set_aux_loss):
    """
        post process of dn after output from the transformer
        put the dn part in the dn_meta
    """
    assert dn_meta['pad_size'] > 0
    pad_size = dn_meta['pad_size']

    known_poses, pred_poses = pred_poses[:,:,:pad_size], pred_poses[:,:,pad_size:]
    known_betas, pred_betas = pred_betas[:,:,:pad_size], pred_betas[:,:,pad_size:]
    known_boxes, pred_boxes = pred_boxes[:,:,:pad_size], pred_boxes[:,:,pad_size:]
    known_confs, pred_confs = pred_confs[:,:,:pad_size], pred_confs[:,:,pad_size:]
    known_j3ds, pred_j3ds = pred_j3ds[:,:,:pad_size], pred_j3ds[:,:,pad_size:]
    known_j2ds, pred_j2ds = pred_j2ds[:,:,:pad_size], pred_j2ds[:,:,pad_size:]
    known_depths, pred_depths = pred_depths[:,:,:pad_size], pred_depths[:,:,pad_size:]

    known_verts, pred_verts = pred_verts[:,:pad_size], pred_verts[:,pad_size:]
    known_transl, pred_transl = pred_transl[:,:pad_size], pred_transl[:,pad_size:]

    out = {'pred_poses': known_poses[-1], 'pred_betas': known_betas[-1],
                'pred_boxes': known_boxes[-1], 'pred_confs': known_confs[-1], 
               'pred_j3ds': known_j3ds[-1], 'pred_j2ds': known_j2ds[-1],
               'pred_depths': known_depths[-1]}
        
    if aux_loss:
        out['aux_outputs'] = _set_aux_loss(known_poses, known_betas,
                                            known_boxes, known_confs,
                                            known_j3ds, known_j2ds, known_depths)

    dn_meta['output_known'] = out


    return pred_poses, pred_betas,\
           pred_boxes, pred_confs,\
           pred_j3ds, pred_j2ds,\
           pred_depths, pred_verts,\
           pred_transl,