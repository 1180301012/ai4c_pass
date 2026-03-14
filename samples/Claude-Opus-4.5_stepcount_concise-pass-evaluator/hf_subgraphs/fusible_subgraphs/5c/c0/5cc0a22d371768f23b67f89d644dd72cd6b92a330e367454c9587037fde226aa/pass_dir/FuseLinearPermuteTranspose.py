import torch
import triton
import triton.language as tl

# Pattern matching function - matches linear followed by permute
def pattern(bias, weight, pos_score):
    tmp_2 = torch.nn.functional.linear(pos_score, weight, bias)
    tmp_3 = tmp_2.permute(0, 3, 1, 2)
    return tmp_3

# Argument extraction function
def replacement_args(bias, weight, pos_score):
    return (bias, weight, pos_score)

# Optimized kernel for linear + permute
# Input: [B, H, W, C=3], Weight: [K, C=3], Bias: [K]
# Output: [B, K, H, W] (permuted layout)
@triton.jit
def linear_permute_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B, H, W, K,
    input_stride_b, input_stride_h, input_stride_w,
    output_stride_b, output_stride_k, output_stride_h, output_stride_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    
    # Decode positions in output [B, K, H, W]
    HW = H * W
    KHW = K * HW
    
    b = offsets // KHW
    remainder = offsets % KHW
    k_idx = remainder // HW
    remainder = remainder % HW
    h = remainder // W
    w = remainder % W
    
    # Load bias
    acc = tl.load(bias_ptr + k_idx, mask=mask, other=0.0)
    
    # Linear computation with C=3 (unrolled)
    C = 3
    
    # Compute base input offset
    input_base = b * input_stride_b + h * input_stride_h + w * input_stride_w
    weight_base = k_idx * C
    
    # Channel 0
    input_val_0 = tl.load(input_ptr + input_base + 0, mask=mask, other=0.0)
    weight_val_0 = tl.load(weight_ptr + weight_base + 0, mask=mask, other=0.0)
    acc += input_val_0 * weight_val_0
    
    # Channel 1
    input_val_1 = tl.load(input_ptr + input_base + 1, mask=mask, other=0.0)
    weight_val_1 = tl.load(weight_ptr + weight_base + 1, mask=mask, other=0.0)
    acc += input_val_1 * weight_val_1
    
    # Channel 2
    input_val_2 = tl.load(input_ptr + input_base + 2, mask=mask, other=0.0)
    weight_val_2 = tl.load(weight_ptr + weight_base + 2, mask=mask, other=0.0)
    acc += input_val_2 * weight_val_2
    
    # Store in permuted layout [B, K, H, W]
    output_offset = b * output_stride_b + k_idx * output_stride_k + h * output_stride_h + w * output_stride_w
    tl.store(output_ptr + output_offset, acc, mask=mask)

@torch.fx.wrap
def fused_linear_permute(bias, weight, pos_score):
    # Get shapes
    B, H, W, C = pos_score.shape
    K = weight.shape[0]
    
    # Move weight and bias to device if needed
    if weight.device != pos_score.device:
        weight = weight.to(pos_score.device)
    if bias.device != pos_score.device:
        bias = bias.to(pos_score.device)
    
    # Output for linear + permute: [B, K, H, W]
    permuted_out = torch.empty((B, K, H, W), device=pos_score.device, dtype=pos_score.dtype)
    
    # Launch linear + permute kernel
    n_elements = B * K * H * W
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    linear_permute_kernel[grid](
        pos_score, weight, bias, permuted_out,
        B, H, W, K,
        pos_score.stride(0), pos_score.stride(1), pos_score.stride(2),
        permuted_out.stride(0), permuted_out.stride(1), permuted_out.stride(2), permuted_out.stride(3),
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return permuted_out

# Replacement function - must return the function reference
def replacement_func():
    return fused_linear_permute