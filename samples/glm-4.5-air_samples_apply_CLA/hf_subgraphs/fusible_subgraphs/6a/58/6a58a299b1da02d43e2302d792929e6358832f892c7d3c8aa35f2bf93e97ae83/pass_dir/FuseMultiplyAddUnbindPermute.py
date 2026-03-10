import torch
import triton
import triton.language as tl

# Pattern matching function for complete computation - match multiply + add + unbind + permute
def pattern(in_0, in_1, in_2):
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return tmp_6, tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_1, in_2)  # Only pass the operands for multiplication

def replacement_func():
    return simple_multiply_kernel_wrapper

# Simple Triton kernel for multiplication
@triton.jit
def simple_multiply_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    x_shape: tl.constexpr,
    y_shape: tl.constexpr,
    output_shape: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    # Calculate offset for this program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < output_shape[-1]  # Assuming last dimension size
    
    # Load operands and multiply
    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)
    out = x * y
    
    # Store result
    tl.store(output_ptr + offset, out, mask=mask)

@torch.fx.wrap
def simple_multiply_kernel_wrapper(x, y):
    # Determine output shape
    output_shape = x.shape  # Assuming broadcasting handles the shape
    output_size = x.numel()
    
    # Choose block size optimally
    BLOCK_SIZE = min(1024, output_size)
    num_programs = (output_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Handle broadcasting reshape and multiply
    # For simplicity, we'll reshape to 1D Triton operations
    x_flat = x.contiguous().view(-1)
    y_flat = y.contiguous().view(-1)
    output_flat = output.contiguous().view(-1)
    
    # Launch kernel
    simple_multiply_kernel[(num_programs,)](
        x_ptr=x_flat,
        y_ptr=y_flat,
        output_ptr=output_flat,
        x_shape=x.shape,
        y_shape=y.shape,
        output_shape=output_shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Triton kernel that handles element-wise operations and handles unbind via kernel launch
@triton.jit
def elementwise_kernel(
    in0_ptr,  # [2, 128] - base tensor
    in1_ptr,  # [1, 1, 2, 128] - multiplication factor
    in2_ptr,  # [N, 17, 1, 128] - main tensor
    
    out_ptr,  # [N, 17, 2, 128] - intermediate result
    
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    unbind_dim_size: tl.constexpr,  # This will be 2
    hidden_size: tl.constexpr,
    
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which batch element, sequence element, and unbind slice this thread handles
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    unbind_slice_id = tl.program_id(2)  # 0 or 1 for the two slices from unbind
    
    # Within each program, handle a block of hidden features
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Calculate base pointers for this batch, sequence, and unbind slice
    # in2: [N, 17, 1, 128] -> access [batch_id, seq_id, 0, :]
    in2_base = in2_ptr + batch_id * seq_len * hidden_size + seq_id * hidden_size
    
    # in1: [1, 1, 2, 128] -> access [0, 0, unbind_slice_id, :]
    in1_base = in1_ptr + unbind_slice_id * hidden_size
    
    # Load data
    in2_vals = tl.load(in2_base + offsets, mask=mask, other=0.0)
    in1_vals = tl.load(in1_base + offsets, mask=mask, other=0.0)
    in0_vals = tl.load(in0_ptr + offsets + unbind_slice_id * hidden_size, mask=mask, other=0.0)
    
    # Perform fused operation: (in2 * in1) + in0
    result = (in2_vals * in1_vals) + in0_vals
    
    # Store result: [N, 17, 2, 128] -> compute location including unbind_slice_id
    out_base = out_ptr + batch_id * seq_len * unbind_dim_size * hidden_size + seq_id * unbind_dim_size * hidden_size + unbind_slice_id * hidden_size
    tl.store(out_base + offsets, result, mask=mask)

# Triton kernel that performs the full fused operation including unbind and permute
@triton.jit
def fused_multiply_add_unbind_permute_full_kernel(
    in0_ptr,  # [2, 128] - base tensor
    in1_ptr,  # [1, 1, 2, 128] - multiplication factor  
    in2_ptr,  # [N, 17, 1, 128] - main tensor
    
    out0_ptr,  # [N, 128, 17] - permuted second slice
    out1_ptr,  # [N, 17, 128] - first slice
    
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    hidden_size: tl.constexpr,
    
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID determines which batch, sequence, and hidden feature this thread handles
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    hidden_offset = tl.program_id(2) * BLOCK_SIZE
    
    # Within each program, handle BLOCK_SIZE hidden features
    offsets = hidden_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    
    # Load data for the first slice (dim=0) and second slice (dim=1)
    # For the first slice (dim=0), we use dim=0 from in2 and dim=0 from in1
    in2_slice0 = tl.load(in2_ptr + batch_id * seq_len * hidden_size + seq_id * hidden_size + offsets, mask=mask, other=0.0)
    in1_slice0 = tl.load(in1_ptr + 0 * hidden_size + offsets, mask=mask, other=0.0)  # dim=0 slice from in1
    in0_slice0 = tl.load(in0_ptr + 0 * hidden_size + offsets, mask=mask, other=0.0)  # first row of in0
    
    # For the second slice (dim=1), we use dim=0 from in2 and dim=1 from in1  
    in2_slice1 = in2_slice0  # Same in2 data, different in1 slice
    in1_slice1 = tl.load(in1_ptr + 1 * hidden_size + offsets, mask=mask, other=0.0)  # dim=1 slice from in1
    in0_slice1 = tl.load(in0_ptr + 1 * hidden_size + offsets, mask=mask, other=0.0)  # second row of in0
    
    # Compute both slices
    result_slice0 = (in2_slice0 * in1_slice0) + in0_slice0
    result_slice1 = (in2_slice1 * in1_slice1) + in0_slice1
    
    # Store first slice result in out1: [batch_size, seq_len, hidden_size]
    out1_base = out1_ptr + batch_id * seq_len * hidden_size + seq_id * hidden_size + offsets
    tl.store(out1_base, result_slice0, mask=mask)
    
    # Store transposed second slice in out0: [batch_size, hidden_size, seq_len] 
    # We need to permute from [batch_size, seq_len, hidden_size] to [batch_size, hidden_size, seq_len]
    out0_base = out0_ptr + batch_id * hidden_size * seq_len + offsets * seq_len + seq_id
    tl.store(out0_base, result_slice1, mask=mask)

# Simple kernel wrapper for fused multiply-add-bind-permute operation  
@torch.fx.wrap
def fused_multiply_add_unbind_permute_kernel_wrapper(in0, in1, in2):
    # Determine input shapes
    batch_size = in2.shape[0]
    seq_len = in2.shape[1]
    hidden_size = in2.shape[3]
    
    # Output shapes
    out0_shape = (batch_size, hidden_size, seq_len)  # permuted result
    out1_shape = (batch_size, seq_len, hidden_size)  # first unbind result
    
    # Create output tensors
    out0 = torch.empty(out0_shape, dtype=in0.dtype, device=in0.device)
    out1 = torch.empty(out1_shape, dtype=in0.dtype, device=in0.device)
    
    # Choose block size optimally based on hidden size
    BLOCK_SIZE = 128
    if hidden_size <= 64:
        BLOCK_SIZE = 64
    elif hidden_size <= 128:
        BLOCK_SIZE = 128
    elif hidden_size <= 256:
        BLOCK_SIZE = 256
    elif hidden_size <= 512:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid_x = batch_size
    grid_y = seq_len  
    grid_z = (hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch fused kernel
    fused_multiply_add_unbind_permute_full_kernel[(grid_x, grid_y, grid_z)](
        in0_ptr=in0,
        in1_ptr=in1,
        in2_ptr=in2,
        out0_ptr=out0,
        out1_ptr=out1,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out0, out1