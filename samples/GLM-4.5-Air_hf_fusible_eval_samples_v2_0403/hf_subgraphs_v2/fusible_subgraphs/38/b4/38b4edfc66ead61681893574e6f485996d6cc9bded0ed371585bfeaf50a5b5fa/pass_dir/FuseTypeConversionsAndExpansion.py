import torch
import triton
import triton.language as tl

@triton.jit
def invert_mask_kernel(mask_ptr, inverted_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    inverted = tl.load(mask_ptr + offsets, mask=mask, other=0.0)
    inverted = 1.0 - inverted
    inverted_bool = inverted > 0.5  # Convert to boolean
    inverted = tl.where(inverted_bool, -3.4028234663852886e+38, inverted)
    
    tl.store(inverted_ptr + offsets, inverted, mask=mask)

@torch.fx.wrap
def process_and_invert_attention_mask(input_tensor, target_shape=(1, 1, 9, 9)):
    """Process input attention mask by expanding, converting to float32, and inverting"""
    n = input_tensor.shape[1]  # Get the sequence length from input
    
    # Slice and expand the input tensor
    sliced_tensor = input_tensor[(slice(None), None, None, slice(None))]
    expanded_tensor = sliced_tensor.expand(target_shape[0], target_shape[1], n, n)
    
    # Convert to float32
    expanded_tensor = expanded_tensor.to(torch.float32)
    
    # Create inverted mask (1.0 - input)
    inverted = torch.full_like(expanded_tensor, 1.0, dtype=torch.float32) - expanded_tensor
    
    # Convert to boolean for masking
    inverted_bool = inverted.bool()
    
    # Apply masked fill with masked_fill operation
    result = inverted.masked_fill(inverted_bool, -3.4028234663852886e+38)
    
    # Final boolean conversion (though this might be redundant based on the original pattern)
    # Keep it for safety to match original behavior exactly
    result = result.bool().to(torch.float32)
    
    return result

def pattern(a, b):
    """Pattern to match the type conversions and expansion operations"""
    # This matches: tmp_10 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_10 = a[(slice(None, None, None), None, None, slice(None, None, None))]
    # tmp_11 = tmp_10.expand(1, 1, 9, 9)
    tmp_11 = tmp_10.expand(1, 1, 9, 9)
    # tmp_12 = tmp_11.to(torch.float32)
    tmp_12 = tmp_11.to(torch.float32)
    # tmp_13 = torch.tensor(1.0, dtype = torch.float32)
    tmp_13 = torch.tensor(1.0, dtype=torch.float32)
    # tmp_14 = tmp_13 - tmp_12
    tmp_14 = tmp_13 - tmp_12
    # tmp_15 = tmp_14.to(torch.bool)
    tmp_15 = tmp_14.to(torch.bool)
    # tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_16 = tmp_14.masked_fill(tmp_15, -3.4028234663852886e+38)
    # tmp_17 = tmp_16.to(device(type='cuda', index=0))
    tmp_17 = tmp_16.to(torch.device(type='cuda', index=0))
    # tmp_18 = tmp_17.bool()
    tmp_18 = tmp_17.bool()
    return tmp_18

def replacement_args(a, b):
    """Extract arguments for the replacement function"""
    return (a, b)

def replacement_func():
    return process_and_invert_attention_mask