import torch
import triton
import triton.language as tl

# Pattern matching function - matches: linear + add + relu_
def pattern(in_0, in_1, in_2, in_3):
    # in_0: bias [128]
    # in_1: weight [128, 128]
    # in_2: tensor to add [1000, 128]
    # in_3: input [1000, 128]
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_3, tmp_1, tmp_0)
    tmp_3 = in_2 + tmp_2
    tmp_4 = tmp_3.relu_()
    return tmp_4

# Argument extraction function
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Optimized kernel: Add + ReLU with multiple rows per program
# This reduces kernel launch overhead while maintaining good occupancy
@triton.autotune(
    configs=[
        # Config: BLOCK_M (rows per program), BLOCK_N (elements per row), num_warps
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_warps': 8}, num_stages=1),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_warps': 8}, num_stages=1),
    ],
    key=['M', 'N'],
)
@triton.jit
def fused_add_relu_kernel(
    input_ptr, add_ptr, output_ptr,
    M, N,
    stride_in, stride_io,
    stride_an, stride_ao,
    stride_on, stride_oo,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """
    Fused kernel: output = ReLU(input + add)
    Each program processes BLOCK_M rows x BLOCK_N columns.
    """
    # Get program id
    pid = tl.program_id(0)
    
    # Calculate row and col offsets
    row_offset = pid * BLOCK_M
    col_offset = 0
    
    # Create row and col indices
    rows = row_offset + tl.arange(0, BLOCK_M)
    cols = col_offset + tl.arange(0, BLOCK_N)
    
    # Create masks
    row_mask = rows < M
    col_mask = cols < N
    
    # Load input block
    input_ptrs = input_ptr + rows[:, None] * stride_in + cols[None, :] * stride_io
    input_mask = row_mask[:, None] & col_mask[None, :]
    input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)
    
    # Load add block
    add_ptrs = add_ptr + rows[:, None] * stride_an + cols[None, :] * stride_ao
    add_mask = row_mask[:, None] & col_mask[None, :]
    add_vals = tl.load(add_ptrs, mask=add_mask, other=0.0)
    
    # Add + ReLU (fused)
    result = input_vals + add_vals
    result = tl.where(result > 0, result, 0.0)
    
    # Store output
    output_ptrs = output_ptr + rows[:, None] * stride_on + cols[None, :] * stride_oo
    tl.store(output_ptrs, result, mask=input_mask)


@torch.fx.wrap
def fused_add_relu_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper for Add + ReLU fusion with multi-row processing.
    
    Args:
        in_0: bias [128]
        in_1: weight [128, 128]
        in_2: tensor to add [1000, 128]
        in_3: input [1000, 128]
    
    Returns:
        ReLU(in_3 @ in_1.t() + in_0 + in_2)
    """
    M, N = in_3.shape  # [1000, 128]
    
    # Do linear operation using PyTorch (optimized cuBLAS)
    linear_out = in_3 @ in_1.t() + in_0  # [1000, 128]
    
    # Prepare inputs
    linear_out_f = linear_out.contiguous()
    add_tensor = in_2.contiguous()
    
    # Allocate output
    output = torch.empty((M, N), device=linear_out_f.device, dtype=linear_out_f.dtype)
    
    # Calculate grid - process multiple rows per program
    num_pid_m = triton.cdiv(M, 8)  # BLOCK_M=8 from first config
    grid = (num_pid_m,)
    
    # Launch kernel
    fused_add_relu_kernel[grid](
        linear_out_f, add_tensor, output,
        M, N,
        linear_out_f.stride(0), linear_out_f.stride(1),
        add_tensor.stride(0), add_tensor.stride(1),
        output.stride(0), output.stride(1),
    )
    
    return output


def replacement_func():
    return fused_add_relu_wrapper