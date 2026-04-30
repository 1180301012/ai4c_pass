import torch
import triton
import triton.language as tl


@triton.jit
def _fused_emb_exp_kernel(
    weight_ptr, indices_ptr, output_ptr,
    D: tl.constexpr, SS: tl.constexpr,
    B: tl.constexpr, BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    ij = pid * BLOCK + tl.arange(0, BLOCK)
    mask = ij < SS

    idx = tl.load(indices_ptr + ij, mask=mask, other=0).to(tl.int32)
    base_w = idx * D

    for d in range(D):
        val = tl.load(weight_ptr + base_w + d, mask=mask, other=0.0)
        for b in range(B):
            tl.store(output_ptr + (b * D + d) * SS + ij, val, mask=mask)


@torch.fx.wrap
def fused_embedding_expand_dispatch(weight, indices, route):
    D = weight.shape[1]

    if route == "r_1_45_45":
        S, B, BLOCK, nw = 45, 1, 2048, 4
    elif route == "r_1_11_11":
        S, B, BLOCK, nw = 11, 1, 128, 1
    elif route == "r_2_7_7":
        S, B, BLOCK, nw = 7, 2, 64, 1
    else:
        S, B, BLOCK, nw = 45, 1, 2048, 4

    SS = S * S
    output = torch.empty((B, D, S, S), dtype=weight.dtype, device=indices.device)

    _fused_emb_exp_kernel[(1,)](
        weight, indices, output,
        D=D, SS=SS, B=B, BLOCK=BLOCK,
        num_warps=nw,
    )

    return output