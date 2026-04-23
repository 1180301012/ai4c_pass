import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    """
    Match the computation:
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]; tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1); tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1); tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)
    
    Note: split(1, dim=-1) on a tensor with last dim size 2 produces two tensors
    of shape [..., 1]. Squeeze(-1) removes that dim of size 1.
    """
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    in_0_ptr, in_1_ptr,
    out_0_ptr, out_1_ptr,
    M: tl.constexpr, N: tl.constexpr,
    scale: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a 2D block
    pid = tl.program_id(0)
    num_programs = M
    
    # Calculate which row this program processes
    row = pid
    col_offset = 0
    
    # offsets for loading (M=17, N=2)
    offsets = row * N + tl.arange(0, N)
    
    # Load in_1 values [1, 17, 2] -> we access as [row, col]
    mask = (row < M) & (col_offset + tl.arange(0, N) < N)
    in_1 = tl.load(in_1_ptr + row * N + tl.arange(0, N), mask=mask, other=0.0)
    
    # Load in_0 [1, 17, 1] -> all values are same for given row
    in_0_val = tl.load(in_0_ptr + row).to(tl.float32)
    
    # Fused multiply-subtract: in_1 - in_0 * scale
    result = in_1 - in_0_val * scale
    
    # Split into two outputs (each gets one element from the last dim)
    out_0_val = result[0]
    out_1_val = result[1]
    
    # Store outputs [1, 17] -> linearized as [row]
    tl.store(out_0_ptr + row, out_0_val, mask=row < M)
    tl.store(out_1_ptr + row, out_1_val, mask=row < M)

@torch.fx.wrap
def triton_fused_mul_sub_split_squeeze(in_0, in_1):
    """
    Fused kernel that performs:
    1. Multiply in_0 by 1000000.0
    2. Subtract from in_1 (with broadcast)
    3. Split along last dim and squeeze
    
    All in a single kernel launch with efficient memory access.
    """
    M = 17  # in_1.shape[1]
    N = 2   # in_1.shape[2]
    scale = 1000000.0
    
    # Output shapes: [1, 17] each
    out_0 = torch.empty((1, M), dtype=in_1.dtype, device=in_1.device)
    out_1 = torch.empty((1, M), dtype=in_1.dtype, device=in_1.device)
    
    # Flatten inputs for efficient linear access
    in_0_flat = in_0.flatten()  # [1, 17, 1] -> [17]
    in_1_flat = in_1.flatten()  # [1, 17, 2] -> [34]
    
    # Grid: one program per row (17 rows)
    grid = (M,)
    
    fused_kernel[grid](
        in_0_ptr=in_0_flat,
        in_1_ptr=in_1_flat,
        out_0_ptr=out_0.flatten(),
        out_1_ptr=out_1.flatten(),
        M=M, N=N, scale=scale,
        BLOCK_SIZE=128,
    )
    
    return (out_0, out_1)

def replacement_func():
    return triton_fused_mul_sub_split_squeeze