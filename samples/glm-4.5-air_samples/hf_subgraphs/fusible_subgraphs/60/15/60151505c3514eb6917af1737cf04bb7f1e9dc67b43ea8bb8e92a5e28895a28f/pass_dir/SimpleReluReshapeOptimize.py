import torch
import triton
import triton.language as tl

# Pattern matching function - matches ReLU + Reshape operations
def pattern(in_0, in_1):
    # Apply ReLU to in_1
    relu_out = torch.nn.functional.relu(in_1, inplace=True)
    # Reshape in_0
    reshaped_in0 = in_0.reshape(-1, 256, -1)
    # Reshape ReLU output to match
    reshaped_relu = relu_out.reshape(-1, 256, -1)
    # Return both (matches the original computation structure)
    return (reshaped_relu.permute(0, 2, 1), reshaped_in0)

def replacement_args(in_0, in_1):
    # Extract batch size and h_size from input shapes
    batch_size = in_0.shape[0]
    h_size = in_0.shape[2]  # This is the H dimension (21 or 19)
    return (in_0, in_1, batch_size, h_size)

@triton.jit
def simple_fused_kernel(
    # Input pointers
    in_0_ptr, 
    in_1_ptr,
    # Output pointers  
    out_0_ptr,  # permuted result
    out_1_ptr,  # reshaped in_0
    # Shape information
    batch_size: tl.constexpr,
    h_size: tl.constexpr,  # H dimension (21 or 19)
    # Block size
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Each program handles one row of the output [batch_size, h_size, 256]
    if pid >= batch_size * h_size:
        return
    
    # Calculate offsets 
    batch_idx = pid // h_size
    row_idx = pid % h_size
    
    # Process each row (256 elements)
    start_offset = batch_idx * h_size * 256 + row_idx * 256
    offsets = start_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_idx * h_size * 256 + (row_idx + 1) * 256
    
    # Process in_0: Reshape from [batch_size, 256, h_size, 1] to [batch_size, h_size, 256]
    orig_offset = batch_idx * h_size * 256 + row_idx * 256
    orig_offsets = orig_offset + tl.arange(0, BLOCK_SIZE)
    orig_mask = orig_offsets < orig_offset + 256
    
    in_0_data = tl.load(in_0_ptr + orig_offsets, mask=orig_mask, other=0.0)
    tl.store(out_1_ptr + offsets, in_0_data, mask=mask)
    
    # Process in_1: ReLU + Reshape + Permute
    relu_data = tl.load(in_1_ptr + orig_offsets, mask=orig_mask, other=0.0)
    relu_out = tl.maximum(relu_data, 0.0)
    tl.store(out_0_ptr + offsets, relu_out, mask=mask)

@torch.fx.wrap
def simple_fused_wrapper(in_0, in_1):
    batch_size = in_0.shape[0]
    h_size = in_0.shape[2]  # H dimension (21 or 19)
    
    # Create output tensors
    out_0 = torch.empty((batch_size, h_size, in_0.shape[1]), dtype=in_1.dtype, device=in_1.device)
    out_1 = torch.empty((batch_size, in_0.shape[1], h_size), dtype=in_0.dtype, device=in_0.device)
    
    BLOCK_SIZE = 256
    num_programs = batch_size * h_size
    
    # Launch kernel
    simple_fused_kernel[(num_programs, 1)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_0_ptr=out_0,
        out_1_ptr=out_1,
        batch_size=batch_size,
        h_size=h_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out_0, out_1

def replacement_func():
    return simple_fused_wrapper