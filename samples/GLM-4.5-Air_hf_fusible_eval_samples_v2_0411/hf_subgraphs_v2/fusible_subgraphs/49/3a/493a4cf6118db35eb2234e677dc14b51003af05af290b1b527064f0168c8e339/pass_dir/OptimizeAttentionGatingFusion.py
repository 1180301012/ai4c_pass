import torch
import triton
import triton.language as tl

def pattern(softmax_weights, gating_param_reshaped, patch_scores):
    """
    Matches the redundant sigmoid computation pattern.
    This matches the computation after softmax is already computed.
    
    Original pattern that this replaces:
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = 1.0 - tmp_3
    tmp_5 = tmp_4 * in_1
    tmp_6 = torch.sigmoid(tmp_2)  # REDUNDANT - same as tmp_3
    tmp_7 = tmp_6 * softmax_weights
    tmp_8 = tmp_5 + tmp_7
    """
    # First computation branch
    sigmoid_val = torch.sigmoid(gating_param_reshaped)
    sigmoid_complement = 1.0 - sigmoid_val
    branch1 = sigmoid_complement * patch_scores
    
    # Second computation branch (recomputes sigmoid unnecessarily)
    sigmoid_val2 = torch.sigmoid(gating_param_reshaped)  # REDUNDANT
    branch2 = sigmoid_val2 * softmax_weights
    
    # Combine branches
    result = branch1 + branch2
    return result

def replacement_args(in_0, in_1, in_2):
    """
    Extract the intermediate values that match our pattern.
    The system will provide these as intermediate tensor values during pattern matching.
    """
    # The pattern matches with these intermediate values computed by the system:
    # softmax_weights = in_2.softmax(dim=-1)  (tmp_1 in original)
    # gating_param_reshaped = in_0.view(1, -1, 1, 1)  (tmp_2 in original)  
    # patch_scores = in_1  (used directly in original)
    
    # Since the pattern matching provides these as arguments, we just return them
    return (in_1, in_0.view(1, -1, 1, 1), in_2.softmax(dim=-1))

@triton.jit
def attention_gating_kernel(
    patch_scores_ptr,      # patch_scores: [1, 16, 196, 196] patch scores
    gating_param_reshaped_ptr,  # gating_param_reshaped: [1, 16, 1, 1] gating parameter
    softmax_ptr,          # softmax_weights: [1, 16, 196, 196] softmax of pos_scores
    output_ptr,            # Output: [1, 16, 196, 196]
    
    # Shapes and strides
    channels: tl.constexpr,        # 16  
    height: tl.constexpr,          # 196
    width: tl.constexpr,           # 196
    spatial_size: tl.constexpr,    # height * width = 196 * 196
    gating_dim: tl.constexpr,      # 16
    
    # Data type handling  
    input_dtype: tl.constexpr,
):
    """
    Optimized Triton kernel that eliminates redundant sigmoid computation.
    Computes sigmoid once and reuses it for both branches.
    """
    # Program IDs: each program handles one spatial position in one channel
    pid_c = tl.program_id(0)  # Channel dimension  
    pid_s = tl.program_id(1)  # Spatial position (flattened)
    
    # Calculate spatial coordinates
    pid_h = pid_s // width
    pid_w = pid_s % width
    
    # Load gating parameter for current channel from the reshaped tensor
    # The gating_param_reshaped has shape [1, 16, 1, 1], so we index by channel
    gating_idx = pid_c  # Index into the channel dimension [16]
    gating_param = tl.load(gating_param_reshaped_ptr + gating_idx)
    
    # Compute sigmoid with proper type handling (done ONCE, not twice like in original)
    if input_dtype == tl.float16:
        gating_param_fp32 = gating_param.to(tl.float32)
        sigmoid_val_fp32 = 1.0 / (1.0 + tl.exp(-gating_param_fp32))
        sigmoid_val = sigmoid_val_fp32.to(tl.float16)
    elif input_dtype == tl.bfloat16:
        gating_param_fp32 = gating_param.to(tl.float32)  
        sigmoid_val_fp32 = 1.0 / (1.0 + tl.exp(-gating_param_fp32))
        sigmoid_val = sigmoid_val_fp32.to(tl.bfloat16)
    else:  # fp32
        sigmoid_val = 1.0 / (1.0 + tl.exp(-gating_param))
    
    sigmoid_complement = 1.0 - sigmoid_val
    
    # Load patch scores and softmax weights for current channel and spatial position
    patch_score = tl.load(patch_scores_ptr + pid_c * spatial_size + pid_s)
    attention_weight = tl.load(softmax_ptr + pid_c * spatial_size + pid_s)
    
    # Compute both branches using the SAME sigmoid value (no redundant computation)
    branch1 = sigmoid_complement * patch_score
    branch2 = sigmoid_val * attention_weight
    
    # Combine branches
    output_val = branch1 + branch2
    
    # Store result
    output_idx = pid_c * spatial_size + pid_s
    tl.store(output_ptr + output_idx, output_val)

@torch.fx.wrap  
def fused_attention_gating(patch_scores, gating_param_reshaped, softmax_weights):
    """
    Optimized kernel that eliminates redundant sigmoid computation.
    Computes sigmoid once and reuses it for both attention branches.
    """
    # Get tensor shapes and properties
    channels = patch_scores.shape[1]    # 16  
    height = patch_scores.shape[2]      # 196
    width = patch_scores.shape[3]       # 196
    spatial_size = height * width       # height * width = 196 * 196
    gating_dim = gating_param_reshaped.shape[1]  # Should be 16
    
    # Determine data type and map to Triton types
    if patch_scores.dtype == torch.float16:
        input_dtype = tl.float16
    elif patch_scores.dtype == torch.bfloat16:
        input_dtype = tl.bfloat16
    else:
        input_dtype = tl.float32
    
    # Create output tensor on same device and with same dtype as inputs
    output = torch.empty_like(patch_scores)
    
    # Calculate grid dimensions (2D: channels, flattened spatial positions)
    grid = (
        channels,   # Process each channel independently
        spatial_size  # Process each spatial position in parallel
    )
    
    # Launch the optimized kernel (sigmoid computed once, not twice)
    # Note: gating_param_reshaped is passed as-is, kernel handles indexing
    attention_gating_kernel[grid](
        patch_scores_ptr=patch_scores,
        gating_param_reshaped_ptr=gating_param_reshaped,
        softmax_ptr=softmax_weights,
        output_ptr=output,
        channels=channels,
        height=height,
        width=width,
        spatial_size=spatial_size,
        gating_dim=gating_dim,
        input_dtype=input_dtype
    )
    
    return output

def replacement_func():
    return fused_attention_gating