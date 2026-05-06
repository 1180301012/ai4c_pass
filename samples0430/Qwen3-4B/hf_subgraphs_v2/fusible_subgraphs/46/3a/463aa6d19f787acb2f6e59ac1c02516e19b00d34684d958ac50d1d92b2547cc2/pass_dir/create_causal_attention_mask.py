import torch
import triton
import triton.language as tl

def pattern(in_0):
    N = in_0.shape[1]
    idx = torch.arange(0, N, device=in_0.device)
    mask = torch.full((N, N), fill_value=-3.4028234663852886e+38, device=in_0.device)
    mask = torch.triu(mask, diagonal=1)
    mask_bool = idx[None, :] > idx[:, None]
    mask = mask * mask_bool
    mask = mask.unsqueeze(0).unsqueeze(0)
    return mask

def replacement_args(in_0):
    return (in_0,)

def causal_mask_kernel(out_ptr, N, BLOCK_SIZE: tl.constexpr):
    i = tl.program_id(0)
    j = tl.program_id(1)
    i_start = i * BLOCK_SIZE
    j_start = j * BLOCK_SIZE
    i_offset = tl.arange(0, BLOCK_SIZE)
    j_offset = tl.arange(0, BLOCK_SIZE)
    for i_off in i_offset:
        i_idx = i_start + i_off
        for j_off in j_offset:
            j_idx = j_start + j_off
            if i_idx < N and j_idx < N and i_idx > j_idx:
                tl.store(out_ptr + (i_idx * N) + j_idx, -3.4028234663852886e+38)
            # Else: leave as 0 (default value)

@torch.fx.wrap
def kernel_wrapper(in_0):
    N = in_0.shape[1]
    out = torch.empty((1, 1, N, N), device=in_0.device, dtype=torch.float32)
    causal_mask_kernel[(1, 1)](out_ptr=out, N=N, BLOCK_SIZE=128)
    return out

def replacement_func():
    return kernel_wrapper