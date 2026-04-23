import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1 = matmul.reshape(-1, 16, 31)
    tmp_2 = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3 = tmp_2.flatten(1)
    tmp_4 = torch.nn.functional.pad(tmp_3, [0, 15], 'constant', None)
    tmp_5 = tmp_4.reshape(-1, 17, 31)
    tmp_6 = tmp_5[(slice(None, None, None), slice(None, 16, None), slice(15, None, None))]
    tmp_7 = tmp_6.reshape(4, 16, 1, 16, 16)
    tmp_8 = tmp_7.expand(-1, -1, 16, -1, -1)
    tmp_9 = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 256, 256)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "route_256")


# -------------------------------------------------------
# Triton kernel: fully-fused BotNet relative-position attention
# -------------------------------------------------------

@triton.jit
def fused_botnet_attn_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, in_3_ptr, in_4_ptr, out_ptr,
    B, S, D_in, center, dim,
    s0_b, s0_q, s0_k,
    s1_b, s1_0, s1_1, s1_d,
    s2_b, s2_0, s2_1, s2_2, s2_3,
    s3_d, s3_p,
    s4_b, s4_k, s4_d,
    so_b, so_d, so_q,
    S_h: tl.constexpr,
    BLOCK_S1K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DIN: tl.constexpr,
):
    pid = tl.program_id(0)
    b_idx = pid // S
    q_idx = pid % S

    si = q_idx % S_h
    sq = q_idx // S_h

    # Step 1: Compute rpos[s2k] = in_1[b,si,sq,:] @ in_3[:, center+s2k-sq]
    s2k_range = tl.arange(0, S_h)
    d_pos_range = center + s2k_range - sq
    d_pos_valid = d_pos_range >= 0

    rpos = tl.zeros([S_h], dtype=tl.float32)

    for d_in_start in range(0, D_in, BLOCK_DIN):
        d_in_off = d_in_start + tl.arange(0, BLOCK_DIN)
        d_in_mask = d_in_off < D_in

        q_ptr_off = b_idx * s1_b + si * s1_0 + sq * s1_1 + d_in_off * s1_d
        q_vec = tl.load(in_1_ptr + q_ptr_off, mask=d_in_mask, other=0.0)

        w_ptr_off = d_in_off[:, None] * s3_d + d_pos_range[None, :] * s3_p
        w_mask = d_in_mask[:, None] & d_pos_valid[None, :]
        w_tile = tl.load(in_3_ptr + w_ptr_off, mask=w_mask, other=0.0)

        rpos += tl.sum(q_vec[:, None] * w_tile, axis=0)

    # Step 2: Flash attention - grouped by s2k
    d_out_off = tl.arange(0, BLOCK_D)
    d_out_mask = d_out_off < dim

    running_max = tl.full([BLOCK_D], float("-inf"), dtype=tl.float32)
    running_sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    running_out = tl.zeros([BLOCK_D], dtype=tl.float32)

    for s2k_idx in range(S_h):
        rpos_val = rpos[s2k_idx]

        for s1k_start in range(0, S_h, BLOCK_S1K):
            s1k_off = s1k_start + tl.arange(0, BLOCK_S1K)
            s1k_mask = s1k_off < S_h
            k_idx = s2k_idx * S_h + s1k_off
            k_valid = s1k_mask

            # Load in_0[b, q_idx, k_idx]
            in_0_off = b_idx * s0_b + q_idx * s0_q + k_idx * s0_k
            attn_base = tl.load(in_0_ptr + in_0_off, mask=k_valid, other=0.0)

            # Load in_2[b, sq, si, s2k_idx, s1k_off]
            in_2_off = b_idx * s2_b + sq * s2_0 + si * s2_1 + s2k_idx * s2_2 + s1k_off * s2_3
            rel_logits = tl.load(in_2_ptr + in_2_off, mask=k_valid, other=0.0)

            scores = attn_base + rpos_val + rel_logits

            # Online softmax
            block_max = tl.max(scores, axis=0)
            new_max = tl.maximum(running_max, block_max)

            rescale = tl.exp(running_max - new_max)
            running_sum = running_sum * rescale
            running_out = running_out * rescale

            exp_scores = tl.exp(scores - new_max)
            block_sum = tl.sum(exp_scores, axis=0)

            # Load V[b, k_idx, d_out_off]
            v_off = b_idx * s4_b + k_idx[:, None] * s4_k + d_out_off[None, :] * s4_d
            v_mask = k_valid[:, None] & d_out_mask[None, :]
            v_tile = tl.load(in_4_ptr + v_off, mask=v_mask, other=0.0)

            running_out += tl.sum(exp_scores[:, None] * v_tile, axis=0)
            running_sum += block_sum
            running_max = new_max

    # Step 3: Normalize and write output
    final_out = running_out / running_sum
    out_off = b_idx * so_b + d_out_off * so_d + q_idx * so_q
    tl.store(out_ptr + out_off, final_out, mask=d_out_mask)


@torch.fx.wrap
def fused_botnet_attention_dispatch(in_0, in_1, in_2, in_3, in_4, route):
    B = in_0.shape[0]
    dim = in_4.shape[2]

    if route == "route_256":
        S = 256
        S_h = 16
        D_in = 128
        center = 15
    elif route == "route_64":
        S = 64
        S_h = 8
        D_in = 128
        center = 7
    else:
        raise ValueError(f"Unknown route: {route}")

    out = torch.empty((B, dim, S), dtype=in_0.dtype, device=in_0.device)

    s0 = in_0.stride()
    s1 = in_1.stride()
    s2 = in_2.stride()
    s3 = in_3.stride()
    s4 = in_4.stride()
    so = out.stride()

    BLOCK_S1K = S_h
    BLOCK_D = 64
    BLOCK_DIN = 32

    grid = (B * S,)

    fused_botnet_attn_kernel[grid](
        in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2, in_3_ptr=in_3, in_4_ptr=in_4, out_ptr=out,
        B=B, S=S, D_in=D_in, center=center, dim=dim,
        s0_b=s0[0], s0_q=s0[1], s0_k=s0[2],
        s1_b=s1[0], s1_0=s1[1], s1_1=s1[2], s1_d=s1[3],
        s2_b=s2[0], s2_0=s2[1], s2_1=s2[2], s2_2=s2[3], s2_3=s2[4],
        s3_d=s3[0], s3_p=s3[1],
        s4_b=s4[0], s4_k=s4[1], s4_d=s4[2],
        so_b=so[0], so_d=so[1], so_q=so[2],
        S_h=S_h,
        BLOCK_S1K=BLOCK_S1K,
        BLOCK_D=BLOCK_D,
        BLOCK_DIN=BLOCK_DIN,
    )

    return out


def replacement_func():
    return fused_botnet_attention_dispatch