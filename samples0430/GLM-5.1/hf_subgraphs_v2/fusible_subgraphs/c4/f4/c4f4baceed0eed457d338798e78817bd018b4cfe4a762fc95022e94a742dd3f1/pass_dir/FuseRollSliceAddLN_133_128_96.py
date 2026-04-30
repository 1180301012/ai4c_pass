import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fused_roll_slice_add_layernorm_kernel


def pattern(in_0, in_1, in_2, in_3):
    tmp_2 = in_3.contiguous()
    tmp_3 = tmp_2.view(-1, 133, 133, 96)
    tmp_4 = torch.roll(tmp_3, shifts=(3, 3), dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, 128, None), slice(None, 128, None), slice(None, None, None))]
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.view(1, 16384, 96)
    tmp_8 = in_2 + tmp_7
    tmp_9 = torch.nn.functional.layer_norm(tmp_8, (96,), in_1, in_0, 1e-05)
    return tmp_8, tmp_9


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@torch.fx.wrap
def fused_roll_slice_add_layernorm_133_128_96(in_0, in_1, in_2, in_3):
    # Make in_3 contiguous (as in original computation)
    in_3_c = in_3.contiguous()
    
    # Shape parameters
    C = 96
    H_4d = 133
    W_4d = 133
    H_cut = 128
    W_cut = 128
    shift = 3
    N = H_cut * W_cut  # 16384
    eps = 1e-05
    
    BLOCK_C = triton.next_power_of_2(C)  # 128
    
    # Allocate output tensors
    out_add = torch.empty((1, N, C), dtype=in_2.dtype, device=in_2.device)
    out_ln = torch.empty((1, N, C), dtype=in_2.dtype, device=in_2.device)
    
    # Flatten tensors for kernel
    in_3_flat = in_3_c.view(-1)
    in_2_flat = in_2.view(-1)
    out_add_flat = out_add.view(-1)
    out_ln_flat = out_ln.view(-1)
    
    grid = (N,)
    
    fused_roll_slice_add_layernorm_kernel[grid](
        in_3_flat, in_2_flat, in_1, in_0,
        out_add_flat, out_ln_flat,
        N, C, H_4d, W_4d, H_cut, W_cut, shift,
        eps,
        BLOCK_C=BLOCK_C,
    )
    
    return out_add, out_ln


def replacement_func():
    return fused_roll_slice_add_layernorm_133_128_96