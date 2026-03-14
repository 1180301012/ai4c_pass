import torch
import triton
import triton.language as tl

# Pattern for H=14, C=512, SHIFT=3 (Graph 1)
def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 14, 14, 512)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4.view(1, 196, 512)
    tmp_6 = in_2 + tmp_5
    tmp_7 = torch.nn.functional.layer_norm(tmp_6, (512,), in_1, in_0, 1e-05)
    return tmp_6, tmp_7

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_roll_add_layernorm_kernel_H14_C512(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_add_ptr,
    out_ln_ptr,
    N,
    BLOCK_C: tl.constexpr,
    H: tl.constexpr,
    SHIFT: tl.constexpr,
):
    C = BLOCK_C
    
    row_idx = tl.program_id(0)
    
    i = row_idx // H
    j = row_idx % H
    
    src_i = (i - SHIFT + H) % H
    src_j = (j - SHIFT + H) % H
    src_row = src_i * H + src_j
    
    offs = tl.arange(0, BLOCK_C)
    x = tl.load(in_3_ptr + src_row * C + offs)
    residual = tl.load(in_2_ptr + row_idx * C + offs)
    
    add_result = residual + x
    tl.store(out_add_ptr + row_idx * C + offs, add_result)
    
    mean = tl.sum(add_result) / C
    diff = add_result - mean
    var = tl.sum(diff * diff) / C
    inv_std = tl.rsqrt(var + 1e-05)
    normalized = diff * inv_std
    
    weight = tl.load(in_1_ptr + offs)
    bias = tl.load(in_0_ptr + offs)
    ln_result = normalized * weight + bias
    
    tl.store(out_ln_ptr + row_idx * C + offs, ln_result)


@torch.fx.wrap
def _fused_kernel_H14_C512(in_0, in_1, in_2, in_3):
    H = 14
    C = 512
    SHIFT = 3
    N = H * H
    
    in_3_contig = in_3.contiguous()
    
    out_add = torch.empty(1, N, C, device=in_2.device, dtype=in_2.dtype)
    out_ln = torch.empty(1, N, C, device=in_2.device, dtype=in_2.dtype)
    
    grid = (N,)
    fused_roll_add_layernorm_kernel_H14_C512[grid](
        in_0, in_1, in_2, in_3_contig,
        out_add, out_ln,
        N,
        BLOCK_C=C,
        H=H,
        SHIFT=SHIFT,
    )
    
    return out_add, out_ln


def fused_roll_add_layernorm_H14_C512(in_0, in_1, in_2, in_3):
    result = _fused_kernel_H14_C512(in_0, in_1, in_2, in_3)
    out_add = result[0]
    out_ln = result[1]
    return out_add, out_ln


def replacement_func():
    return fused_roll_add_layernorm_H14_C512