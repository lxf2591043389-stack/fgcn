import math
import torch
import torch.nn.functional as F


def _dilate_mask(mask, dilation_r):
    out = mask
    for _ in range(dilation_r):
        out = F.max_pool2d(out.float(), kernel_size=3, stride=1, padding=1) > 0
    return out


def tile_scheduler(
    C_init,
    M,
    k_max=32,
    tau_miss=0.01,
    dilation_r=1,
    lam=0.7,
    fill_to_kmax=True,
    adaptive_k=False,
    risk_top_ratio=0.1,
    k_min=4,
):
    """
    C_init: [B,1,48,72]
    M:      [B,1,192,288] hole=1
    return:
      tiles: LongTensor [B,K,2] each (i,j), i in [0..5], j in [0..8]
    """
    M_1_4 = F.avg_pool2d(M.float(), kernel_size=4, stride=4)
    hole_ratio = F.avg_pool2d(M_1_4, kernel_size=8, stride=8)

    R = 1.0 - C_init
    R_tile = F.avg_pool2d(R, kernel_size=8, stride=8)

    T_miss = hole_ratio > tau_miss
    T_miss_ctx = _dilate_mask(T_miss, dilation_r)

    S = lam * hole_ratio + (1.0 - lam) * R_tile

    B = C_init.shape[0]
    if adaptive_k:
        miss_counts = [int(T_miss[b, 0].sum().item()) for b in range(B)]
        nonmiss_counts = [int((~T_miss[b, 0]).sum().item()) for b in range(B)]
        risk_top_ratio = max(0.0, min(1.0, risk_top_ratio))
        k_list = []
        for b in range(B):
            risk_count = int(math.ceil(risk_top_ratio * nonmiss_counts[b]))
            k_i = max(k_min, min(k_max, miss_counts[b] + risk_count))
            k_list.append(k_i)
        k_out = max(k_list) if k_list else max(1, k_min)
    else:
        miss_counts = [int(T_miss_ctx[b, 0].sum().item()) for b in range(B)]
        if fill_to_kmax:
            k_out = k_max
        else:
            k_out = min(k_max, max(miss_counts)) if max(miss_counts) > 0 else 1
        k_list = [k_out for _ in range(B)]

    tiles = torch.zeros((B, k_out, 2), dtype=torch.long, device=C_init.device)

    for b in range(B):
        k_target = k_list[b]
        miss_mask = T_miss_ctx[b, 0]
        s_map = S[b, 0]
        r_map = R_tile[b, 0]

        miss_idx = miss_mask.nonzero(as_tuple=False)
        if miss_idx.numel() > 0:
            miss_scores = s_map[miss_mask]
            if miss_idx.shape[0] > 1:
                order = torch.argsort(miss_scores, descending=True)
                miss_idx = miss_idx[order]

        if miss_idx.shape[0] >= k_target:
            selected = miss_idx[:k_target]
        else:
            selected = miss_idx
            remaining = k_target - selected.shape[0]
            if remaining > 0:
                risk_mask = ~miss_mask
                risk_idx = risk_mask.nonzero(as_tuple=False)
                if risk_idx.numel() > 0:
                    risk_scores = r_map[risk_mask]
                    k_sel = min(remaining, risk_idx.shape[0])
                    _, topk_idx = torch.topk(risk_scores, k_sel)
                    risk_selected = risk_idx[topk_idx]
                    selected = torch.cat([selected, risk_selected], dim=0)

        if selected.shape[0] < k_out:
            if selected.shape[0] == 0:
                selected = torch.zeros((k_out, 2), dtype=torch.long, device=C_init.device)
            else:
                pad = selected[-1:].repeat(k_out - selected.shape[0], 1)
                selected = torch.cat([selected, pad], dim=0)

        tiles[b] = selected[:k_out]

    return tiles
