import torch
import triton
import triton.language as tl

@triton.jit
def pad_kernel(
    x_ptr,
    y_ptr,
    B,
    S,
    H,
    BLOCK_S: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    block_start_s = pid_s * BLOCK_S
    offsets_s = block_start_s + tl.arange(0, BLOCK_S)
    mask_s = offsets_s < S
    
    block_start_h = pid_h * BLOCK_H
    offsets_h = block_start_h + tl.arange(0, BLOCK_H)
    mask_h = offsets_h < H
    
    # For each element in the block
    s_out = offsets_s
    h_out = offsets_h
    
    mask = (s_out < S) & mask_h
    
    x_val = tl.load(x_ptr + (0 * S * H) + (s_out * H) + h_out, mask=mask, other=0.0)
    
    tl.store(y_ptr + (0 * (S+1) * H) + (s_out * H) + h_out, x_val, mask=mask)

@torch.fx.wrap
def pad_kernel_wrapper(x):
    B = x.shape[0]
    S = x.shape[1]
    H = x.shape[2]
    S_out = S + 1
    
    y = torch.empty((B, S_out, H), dtype=x.dtype, device=x.device)
    
    BLOCK_S = 32
    BLOCK_H = 64
    
    num_s = (S + BLOCK_S - 1) // BLOCK_S
    num_h = (H + BLOCK_H - 1) // BLOCK_H
    
    pad_kernel[(num_s, num_h)](
        x,
        y,
        B,
        S,
        H,
        BLOCK_S=BLOCK_S,
        BLOCK_H=BLOCK_H
    )
    
    return y

def pattern(x):
    return torch.nn.functional.pad(x, (0, 0, 0, 1), 'constant', None)

def replacement_args(x):
    return (x,)

def replacement_func():
    return pad_kernel_wrapper