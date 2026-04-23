# Shared kernel module for embedding+permute+expand fusion
import torch
import triton
import triton.language as tl


@triton.jit
def fused_emb_kernel_batch1(
    weight_ptr,
    indices_ptr,
    output_ptr,
    embed_dim,
    H,
    W,
    HW,
    EHW,
    BLOCK_D: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    hw = pid_hw
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < embed_dim
    
    if hw >= HW:
        return
    
    idx = tl.load(indices_ptr + hw)
    val = tl.load(weight_ptr + idx * embed_dim + d_offsets, mask=d_mask, other=0.0)
    
    out_offsets = d_offsets * HW + hw
    tl.store(output_ptr + out_offsets, val, mask=d_mask)


@triton.jit
def fused_emb_kernel_batch2(
    weight_ptr,
    indices_ptr,
    output_ptr,
    embed_dim,
    H,
    W,
    HW,
    EHW,
    BLOCK_D: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    hw = pid_hw
    d_start = pid_d * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < embed_dim
    
    if hw >= HW:
        return
    
    idx = tl.load(indices_ptr + hw)
    val = tl.load(weight_ptr + idx * embed_dim + d_offsets, mask=d_mask, other=0.0)
    
    # Batch 0
    out_offsets0 = d_offsets * HW + hw
    tl.store(output_ptr + out_offsets0, val, mask=d_mask)
    
    # Batch 1
    out_offsets1 = EHW + d_offsets * HW + hw
    tl.store(output_ptr + out_offsets1, val, mask=d_mask)


@torch.fx.wrap
def fused_embedding_permute_expand_dispatch(indices, weight, route):
    embed_dim = weight.shape[1]
    H = indices.shape[0]
    W = indices.shape[1]
    HW = H * W
    EHW = embed_dim * HW

    if route == "route_1_45_45":
        batch = 1
    elif route == "route_1_11_11":
        batch = 1
    elif route == "route_2_7_7":
        batch = 2
    else:
        batch = 1

    output = torch.empty((batch, embed_dim, H, W), dtype=weight.dtype, device=weight.device)

    BLOCK_D = 4
    num_d_programs = (embed_dim + BLOCK_D - 1) // BLOCK_D

    if batch == 1:
        fused_emb_kernel_batch1[(HW, num_d_programs)](
            weight_ptr=weight,
            indices_ptr=indices,
            output_ptr=output,
            embed_dim=embed_dim,
            H=H,
            W=W,
            HW=HW,
            EHW=EHW,
            BLOCK_D=BLOCK_D,
        )
    else:
        fused_emb_kernel_batch2[(HW, num_d_programs)](
            weight_ptr=weight,
            indices_ptr=indices,
            output_ptr=output,
            embed_dim=embed_dim,
            H=H,
            W=W,
            HW=HW,
            EHW=EHW,
            BLOCK_D=BLOCK_D,
        )

    return output


def replacement_func():
    return fused_embedding_permute_expand_dispatch