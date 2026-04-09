import torch
import triton
import triton.language as tl

@triton.jit
def fused_mul_reshape_sum_kernel(
    softmax_ptr,         # [batch_size, 17, 4096] - softmax output
    weight_x_ptr,        # [1, 1, 1, 64] - broadcast weight
    weight_y_ptr,        # [1, 1, 64, 1] - broadcast weight
    out_sum1_ptr,        # [output_batch, 17, 1] - first sum output
    out_sum2_ptr,        # [output_batch, 17, 1] - second sum output
    batch_size: tl.constexpr,
    output_batch: tl.constexpr,
    feat_h: tl.constexpr,
    feat_w: tl.constexpr,
    n_keypoints: tl.constexpr,
    hidden_dim: tl.constexpr,
):
    # Each program handles one output batch dimension (128, 64, 4, 1, 256, or 512)
    pid = tl.program_id(0)
    
    if pid >= output_batch:
        return
        
    # Process all keypoints and spatial locations for this output batch
    for kp_idx in range(n_keypoints):
        for spatial_idx in range(feat_h * feat_w):
            # Calculate global memory offset for softmax data [batch_idx, kp_idx, spatial_pos]
            softmax_offset = (pid * batch_size + kp_idx) * (feat_h * feat_w) + spatial_idx
            
            # Load softmax values for this keypoint
            softmax_val = tl.load(softmax_ptr + softmax_offset)
            
            # Reshape softmax to [feat_h, feat_w] for spatial operations
            h = spatial_idx // feat_w
            w = spatial_idx % feat_w
            
            # Load weight values (will be broadcasted)
            weight_x = tl.load(weight_x_ptr)
            weight_y = tl.load(weight_y_ptr)
            
            # Compute spatial positions for weight access
            # For weight_x: [1, 1, 1, 64] -> broadcast to [feat_h, feat_w]
            weight_x_val = weight_x  # Broadcast across spatial dimensions
            
            # For weight_y: [1, 1, 64, 1] -> needs specific position access
            weight_y_offset = (h * feat_w + w)  # Map spatial position to weight access
            
            # Load weight_y value with proper indexing
            if weight_y_offset < 64 * 1:
                weight_y_val = tl.load(weight_y_ptr + weight_y_offset)
            else:
                weight_y_val = 0.0
            
            # First operation: softmax * weight_x, reshape and sum
            # Result should go to out_sum1[output_batch, 17, 1]
            sum1_val = softmax_val * weight_x_val
            sum1_offset = pid * n_keypoints * hidden_dim + kp_idx * hidden_dim
            
            # Second operation: softmax * weight_y, reshape and sum  
            # Result should go to out_sum2[output_batch, 17, 1]
            sum2_val = softmax_val * weight_y_val
            sum2_offset = pid * n_keypoints * hidden_dim + kp_idx * hidden_dim
            
            # Store partial sums (across hidden dimension)
            # Hidden dimension reduction happens here (we sum across the full hidden dim)
            if hidden_dim > 0:
                # For now, store individual values - final reduction needs kernel adjustment
                tl.store(out_sum1_ptr + sum1_offset, sum1_val)
                tl.store(out_sum2_ptr + sum2_offset, sum2_val)

@torch.fx.wrap
def fused_mul_reshape_sum_parallel_kernel(softmax, weight_x, weight_y, output_batch_size, target_hidden_dim):
    # Get shapes
    batch_size, n_keypoints, total_features = softmax.shape
    feat_h = feat_w = 64  # Fixed 64x64 spatial grid
    hidden_dim = total_features // n_keypoints
    
    # Output tensors for sums
    out_sum1 = torch.zeros((output_batch_size, n_keypoints, hidden_dim), dtype=softmax.dtype, device=softmax.device)
    out_sum2 = torch.zeros((output_batch_size, n_keypoints, hidden_dim), dtype=softmax.dtype, device=softmax.device)
    
    # Launch kernel with grid = output_batch_size
    fused_mul_reshape_sum_kernel[(output_batch_size,)](
        softmax_ptr=softmax,
        weight_x_ptr=weight_x,
        weight_y_ptr=weight_y,
        out_sum1_ptr=out_sum1,
        out_sum2_ptr=out_sum2,
        batch_size=batch_size,
        output_batch=output_batch_size,
        feat_h=feat_h,
        feat_w=feat_w,
        n_keypoints=n_keypoints,
        hidden_dim=hidden_dim,
    )
    
    # Final reduction to get [output_batch, 17, 1] as required
    final_sum1 = torch.sum(out_sum1, dim=-1, keepdim=True)
    final_sum2 = torch.sum(out_sum2, dim=-1, keepdim=True)
    
    return final_sum1, final_sum2, softmax.reshape(-1, 17, 64, 64)  # Return softmax as tmp_3

def pattern(softmax_input, in_0, in_1):
    """
    Pattern matches the computation:
    softmax -> reshape
    then two parallel paths: reshape * in_0 -> reshape -> sum and reshape * in_1 -> reshape -> sum
    """
    tmp_3 = softmax_input.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(-1, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(-1, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    
    return tmp_3, tmp_10

def replacement_args(softmax_input, in_0, in_1):
    return (softmax_input, in_0, in_1)

def replacement_func():
    return fused_mul_reshape_sum_parallel_kernel