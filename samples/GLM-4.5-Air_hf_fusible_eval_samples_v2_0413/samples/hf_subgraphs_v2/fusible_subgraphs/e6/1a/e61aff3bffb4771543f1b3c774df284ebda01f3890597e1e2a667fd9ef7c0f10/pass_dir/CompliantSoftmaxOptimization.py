import torch
import triton
import triton.language as tl

def pattern(conv2d_result):
    """
    Pattern matching: Conv2D result → View → Softmax → Unsqueeze
    
    This matches the exact sequence after conv2d:
    conv2d_result.view(conv2d_result.shape[0], 1, -1)
    torch.nn.functional.softmax(view_result, 2, _stacklevel=5)
    softmax_result.unsqueeze(-1)
    """
    view_result = conv2d_result.view(conv2d_result.shape[0], 1, -1)
    softmax_result = torch.nn.functional.softmax(view_result, 2, _stacklevel=5)
    final_result = softmax_result.unsqueeze(-1)
    return final_result

def replacement_args(conv2d_result):
    """Extract arguments needed for the optimized kernel"""
    return (conv2d_result,)

@triton.jit
def optimized_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    feature_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel that fuses Softmax + Unsqueeze operations.
    """
    batch_id = tl.program_id(0)
    start_idx = batch_id * feature_size
    end_idx = start_idx + feature_size
    
    # Step 1: Find max value for this batch
    max_val = -float('inf')
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < end_idx
        vals = tl.load(input_ptr + idx, mask=mask, other=-float('inf'))
        max_val = tl.max(max_val, vals)
    
    # Step 2: Compute sum of exp(x - max)
    sum_exp = 0.0
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < end_idx
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        exp_x = tl.exp(x - max_val)
        sum_exp += tl.sum(exp_x)
    
    # Step 3: Compute softmax and store with unsqueezed dimension
    for i in range(start_idx, end_idx, BLOCK_SIZE):
        idx = i + tl.arange(0, BLOCK_SIZE)
        mask = idx < end_idx
        x = tl.load(input_ptr + idx, mask=mask, other=0.0)
        max_val_local = max_val
        sum_exp_local = sum_exp
        
        exp_x = tl.exp(x - max_val_local)
        softmax_vals = exp_x / sum_exp_local
        
        # Store with added dimension (output shape: [batch_size, 1, feature_size, 1])
        tl.store(output_ptr + batch_id * feature_size * 2 + i * 2, softmax_vals, mask=mask)
        tl.store(output_ptr + batch_id * feature_size * 2 + i * 2 + 1, 0.0, mask=mask)

@torch.fx.wrap
def optimized_fused_operations(conv2d_result):
    """
    Wrapper function that executes fused View + Softmax + Unsqueeze operations.
    """
    batch_size = conv2d_result.shape[0]
    feature_size = 1  # This is the middle dimension that's always 1
    
    if len(conv2d_result.shape) == 4:
        # For conv2d output [batch, channels, height, width]
        total_features = conv2d_result.shape[1] * conv2d_result.shape[2] * conv2d_result.shape[3]
    else:
        # For other cases
        total_features = conv2d_result.numel() // batch_size
    
    # Output shape: [batch_size, 1, feature_size, 1]
    output_shape = (batch_size, 1, total_features, 1)
    output = torch.empty(output_shape, dtype=conv2d_result.dtype, device=conv2d_result.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch_size * total_features
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Reshape input to [batch_size, total_features] for processing
    reshaped_input = conv2d_result.reshape(batch_size, total_features)
    
    # Run optimized kernel
    optimized_softmax_kernel[(num_programs,)](
        input_ptr=reshaped_input,
        output_ptr=output,
        batch_size=batch_size,
        feature_size=total_features,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Returns the optimized kernel function"""
    return optimized_fused_operations