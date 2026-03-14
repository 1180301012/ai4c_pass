import torch
import triton
import triton.language as tl

# Pattern for H=96, C=128, SHIFT=6 (Graph 4)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 96, 96, 128)
    tmp_4 = torch.roll(tmp_3, shifts=(6, 6), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 9216, 128)
    tmp_6 = in_2 + tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (128,), in_1, in_0, 1e-05)
    return tmp_6, tmp_7

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel - one row per program with better warp usage
@triton.jit
def fused_roll_add_layernorm_kernel(
    in_0_ptr,  # bias
    in_1_ptr,  # weight
    in_2_ptr,  # residual
    in_3_ptr,  # input (rolled)
    out_add_ptr,
    out_ln_ptr,
    N,
    C: tl.constexpr,
    H: tl.constexpr,
    SHIFT: tl.constexpr,
):
    row_idx = tl.program_id(0)
    
    i = row_idx // H
    j = row_idx % H
    
    src_i = (i - SHIFT + H) % H
    src_j = (j - SHIFT + H) % H
    src_row = src_i * H + src_j
    
    offs = tl.arange(0, C)
    
    # Load with roll transformation
    x = tl.load(in_3_ptr + src_row * C + offs)
    residual = tl.load(in_2_ptr + row_idx * C + offs)
    
    # Add
    add_result = residual + x
    
    # Store add result
    tl.store(out_add_ptr + row_idx * C + offs, add_result)
    
    # Layer norm - compute mean
    mean = tl.sum(add_result, axis=0) / C
    
    # Compute variance
    diff = add_result - mean
    var = tl.sum(diff * diff, axis=0) / C
    
    # Normalize
    inv_std = tl.rsqrt(var + 1e-05)
    normalized = diff * inv_std
    
    # Apply affine transformation
    weight = tl.load(in_1_ptr + offs)
    bias = tl.load(in_0_ptr + offs)
    ln_result = normalized * weight + bias
    
    # Store layer norm result
    tl.store(out_ln_ptr + row_idx * C + offs, ln_result)


@torch.fx.wrap
def _fused_kernel_H96_C128(in_0, in_1, in_2, in_3):
    H = 96
    C = 128
    SHIFT = 6
    N = H * H
    
    in_3_contig = in_3.contiguous()
    
    out_add = torch.empty(1, N, C, device=in_2.device, dtype=in_2.dtype)
    out_ln = torch.empty(1, N, C, device=in_2.device, dtype=in_2.dtype)
    
    grid = (N,)
    
    # Use 2 warps for C=128 (each warp handles 32 elements, 2 warps = 64, need 2 iterations)
    fused_roll_add_layernorm_kernel[grid](
        in_0, in_1, in_2, in_3_contig,
        out_add, out_ln,
        N,
        C=C,
        H=H,
        SHIFT=SHIFT,
        num_warps=2,
        num_stages=2,
    )
    
    return out_add, out_ln


def fused_roll_add_layernorm_H96_C128(in_0, in_1, in_2, in_3):
    result = _fused_kernel_H96_C128(in_0, in_1, in_2, in_3)
    out_add = result[0]
    out_ln = result[1]
    return out_add, out_ln


def replacement_func():
    return fused_roll_add_layernorm_H96_C128