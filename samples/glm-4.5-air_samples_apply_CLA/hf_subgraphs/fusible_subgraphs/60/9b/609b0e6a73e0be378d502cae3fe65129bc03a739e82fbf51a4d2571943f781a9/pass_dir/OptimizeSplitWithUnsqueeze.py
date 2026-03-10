import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    # Focus on the key optimization: split + unsqueeze on specific slice
    # This matches the exact computation pattern where the last chunk is unsqueezed
    
    # Extract the specific chunks that correspond to the original split
    # We know the split produces chunks of size [512, 512, 128]
    chunk_512a = input_tensor[:, :, 0:512]
    chunk_512b = input_tensor[:, :, 512:1024]  
    chunk_128 = input_tensor[:, :, 1024:1152]
    
    # The key optimization is the unsqueeze on the 128-element chunk
    chunk_128_expanded = chunk_128.unsqueeze(2)
    
    # Return all three chunks to match the expected output structure
    return chunk_512a, chunk_512b, chunk_128_expanded

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_split_unsqueeze_kernel(
    input_ptr,
    output_chunk0_ptr,
    output_chunk1_ptr, 
    output_chunk2_ptr,
    input_batch,
    input_height,
    input_width,
    chunk0_size,
    chunk1_size,
    chunk2_size,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch and height position
    batch = tl.program_id(0)
    height = tl.program_id(1)
    
    # Calculate input and output strides
    input_stride_batch = input_height * input_width
    input_stride_height = input_width
    
    # Pointers for this batch and height
    input_base = input_ptr + batch * input_stride_batch + height * input_stride_height
    
    # Process each chunk in parallel across width
    offsets = tl.arange(0, BLOCK_SIZE)
    
    # Process chunk 0
    chunk0_start = 0
    chunk0_end = chunk0_size
    chunk0_offsets = chunk0_start + offsets
    chunk0_mask = chunk0_offsets < chunk0_end
    chunk0_data = tl.load(input_base + chunk0_offsets, mask=chunk0_mask, other=0.0)
    tl.store(output_chunk0_ptr + batch * chunk0_size + height * chunk0_size + offsets, 
             chunk0_data, mask=chunk0_mask)
    
    # Process chunk 1  
    chunk1_start = chunk0_size
    chunk1_end = chunk0_size + chunk1_size
    chunk1_offsets = chunk1_start + offsets
    chunk1_mask = chunk1_offsets < chunk1_end
    chunk1_data = tl.load(input_base + chunk1_offsets, mask=chunk1_mask, other=0.0)
    tl.store(output_chunk1_ptr + batch * chunk1_size + height * chunk1_size + offsets,
             chunk1_data, mask=chunk1_mask)
    
    # Process chunk 2 and apply unsqueeze
    chunk2_start = chunk0_size + chunk1_size
    chunk2_end = chunk0_size + chunk1_size + chunk2_size
    chunk2_offsets = chunk2_start + offsets
    chunk2_mask = chunk2_offsets < chunk2_end
    chunk2_data = tl.load(input_base + chunk2_offsets, mask=chunk2_mask, other=0.0)
    
    # For unsqueeze, we need to copy to a new dimension
    # Create output with additional dim at position 2
    output_chunk2_ptr_expanded = output_chunk2_ptr + batch * chunk2_size * 1 + height * chunk2_size * 1
    tl.store(output_chunk2_ptr_expanded + offsets * 1, 
             chunk2_data, mask=chunk2_mask)

@torch.fx.wrap
def optimized_split_unsqueeze(input_tensor):
    # Input shape: [batch, height, 1152] where 1152 = 512+512+128
    batch, height, width = input_tensor.shape
    chunk0_size, chunk1_size, chunk2_size = 512, 512, 128
    
    # Create output tensors
    output_chunk0 = torch.empty((batch, height, chunk0_size), dtype=input_tensor.dtype, device=input_tensor.device)
    output_chunk1 = torch.empty((batch, height, chunk1_size), dtype=input_tensor.dtype, device=input_tensor.device)
    output_chunk2 = torch.empty((batch, height, 1, chunk2_size), dtype=input_tensor.dtype, device=input_tensor.device)  # unsqueeze
    
    BLOCK_SIZE = 1024
    # Grid: (batch, height, 1) - one program per batch and height
    grid = (batch, height, 1)
    
    optimized_split_unsqueeze_kernel[grid](
        input_tensor,
        output_chunk0, 
        output_chunk1,
        output_chunk2,
        batch, height, width,
        chunk0_size, chunk1_size, chunk2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output_chunk0, output_chunk1, output_chunk2

def replacement_func():
    return optimized_split_unsqueeze