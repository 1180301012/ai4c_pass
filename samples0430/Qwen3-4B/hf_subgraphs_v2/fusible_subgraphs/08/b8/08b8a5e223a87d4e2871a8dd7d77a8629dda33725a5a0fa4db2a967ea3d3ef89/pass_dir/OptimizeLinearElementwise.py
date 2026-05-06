import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_0, None)
    tmp_3 = in_2 * in_1
    return (tmp_3, linear)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    return kernel_wrapper

@triton.jit
def optimized_kernel(
    in_3_ptr: tl.tensor,
    in_0_ptr: tl.tensor,
    in_2_ptr: tl.tensor,
    in_1_ptr: tl.tensor,
    out_1_ptr: tl.tensor,
    out_2_ptr: tl.tensor,
    B: tl.int32,
    N: tl.int32,
    C_in: tl.int32,
    C_out: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.arange(0, BLOCK_SIZE)
    
    # Get global indices
    batch_id = tl.program_id(0)  # We process entire batch at once
    seq_id = tl.program_id(1)
    
    # Load data
    in_3_data = tl.load(in_3_ptr + (batch_id, seq_id, row))
    in_0_data = tl.load(in_0_ptr + (row,))
    
    # Matrix multiplication
    linear_out = tl.dot(in_3_data, in_0_data)
    
    # Element-wise multiply
    in_1_data = tl.load(in_1_ptr + (row,))
    out_2_data = linear_out * in_1_data
    
    # Store output
    tl.store(out_1_ptr + (batch_id, seq_id, row), linear_out)
    tl.store(out_2_ptr + (batch_id, seq_id, row), out_2_data)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    B = in_0.shape[0]
    N = in_0.shape[1]
    C_in = in_3.shape[2]
    C_out = in_0.shape[0]
    
    # Create output tensors
    out_1 = torch.empty((B, N, C_out), device=in_0.device, dtype=in_0.dtype)
    out_2 = torch.empty((B, N, C_out), device=in_2.device, dtype=in_2.dtype)
    
    # Launch kernel
    grid = (1, 1, 1)
    optimized_kernel[grid](
        in_3_ptr=in_3,
        in_0_ptr=in_0,
        in_2_ptr=in_2,
        in_1_ptr=in_1,
        out_1_ptr=out_1,
        out_2_ptr=out_2,
        B=B,
        N=N,
        C_in=C_in,
        C_out=C_out,
        BLOCK_SIZE=128,
    )
    
    return (out_2, out_1)