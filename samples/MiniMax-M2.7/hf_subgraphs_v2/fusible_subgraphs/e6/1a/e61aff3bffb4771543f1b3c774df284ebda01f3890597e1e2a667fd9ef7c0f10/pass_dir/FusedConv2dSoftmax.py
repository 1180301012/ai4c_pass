import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv_softmax_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HW: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for 1x1 conv + view + softmax pattern.
    
    Input shape: [B, C, H, W]
    Weight shape: [1, C, 1, 1] (reshaped to [C])
    Bias shape: [1]
    
    The pattern computes:
    1. Pointwise conv: conv[b,c,h,w] = input[b,c,h,w] * weight[c] + bias[c]
    2. View: view[b, 0, hw] = sum_c(conv[b,c,h,w]) where hw = h*W + w
    3. Softmax: softmax[b, 0, hw] = exp(view[b,0,hw]) / sum_j(exp(view[b,0,j]))
    4. Unsqueeze: output[b, 0, hw, 0] = softmax[b, 0, hw]
    
    Grid: One program per (b, h, w) position = B * H * W programs
    """
    # Calculate position (one program per (b, h, w) position)
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate (b, h, w) from program id
    b = pid // (H * W)
    h = (pid % (H * W)) // W
    w = (pid % (H * W)) % W
    
    # Load input at this (b, h, w) position across all channels
    x = tl.load(input_ptr + b * C * H * W + h * W + w + tl.arange(0, BLOCK_SIZE) * H * W, mask=mask)
    
    # Load weight and bias (per-channel)
    w_vals = tl.load(weight_ptr + tl.arange(0, BLOCK_SIZE), mask=mask)
    b_val = tl.load(bias_ptr)
    
    # Apply 1x1 conv: conv = input * weight + bias
    conv_out = x * w_vals + b_val
    
    # Compute sum across channels
    conv_sum = tl.sum(conv_out, axis=0)
    
    # For softmax normalization, we need to compute sum of exp over all HW positions
    # Since each program computes one position, we use a reduction approach
    # We compute exp(conv_sum) / sum_j(exp(conv_sum_j)) for all j in HW
    
    # Store intermediate conv_sum (this is the pre-softmax value)
    # We'll do softmax normalization in a second pass
    
    # Actually, for correct softmax, we need:
    # 1. Find max across all HW positions for numerical stability
    # 2. Compute sum of exp(x - max) across all HW positions
    # 3. Compute exp(x - max) / sum
    
    # For now, let's do a simple version that works for small HW
    # Each program computes its value and we handle the normalization
    
    # Compute exp(conv_sum)
    exp_val = tl.exp(conv_sum)
    
    # Store to intermediate buffer (we'll do reduction)
    tmp_ptr = output_ptr  # Reuse output as intermediate
    tl.store(tmp_ptr + pid, exp_val, mask=mask)


# Second kernel for softmax reduction
@triton.jit  
def softmax_normalize_kernel(
    exp_ptr, output_ptr,
    B: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Normalize by computing softmax over all HW positions for each batch.
    Uses iterative reduction for numerical stability.
    """
    pid = tl.program_id(0)
    b = pid  # One program per batch element
    
    # Load all exp values for this batch
    exp_vals = tl.load(exp_ptr + b * HW + tl.arange(0, BLOCK_SIZE), mask=tl.arange(0, BLOCK_SIZE) < HW)
    
    # Find max for numerical stability
    max_val = tl.max(exp_vals)
    
    # Shift by max and sum
    shifted = tl.exp(tl.log(exp_vals + 1e-10) - max_val)  # log then exp to avoid overflow
    sum_val = tl.sum(shifted)
    
    # Compute softmax
    softmax_out = tl.exp(tl.log(exp_vals + 1e-10) - max_val - tl.log(sum_val + 1e-10))
    
    # Store output (reshape to [B, 1, HW, 1])
    for hw in range(HW):
        tl.store(output_ptr + b * HW + hw, softmax_out[hw], mask=True)
       
    
# A simpler single-pass version that handles softmax inline
@triton.jit
def fused_conv_softmax_single_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HW: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-pass fused kernel for 1x1 conv + view + softmax.
    
    Grid: B * HW programs (one per batch/spatial position)
    
    This kernel:
    1. Computes conv for all C channels at this (b, hw) position
    2. Sums across channels to get pre-softmax value  
    3. Outputs the result (softmax done in separate kernel or skipped for efficiency)
    """
    # Calculate position
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate (b, hw) from program id
    b = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W
    
    # Base offset for this (b, h, w) position
    base_offset = b * C * H * W + h * W + w
    
    # Load all channel values for this spatial position
    # Using loop to handle arbitrary C values
    conv_sum = float(0.0)
    
    # Iteratively compute conv sum across channels
    for c_idx in range(C):
        x = tl.load(input_ptr + base_offset + c_idx * H * W, mask=mask)
        w_val = tl.load(weight_ptr + c_idx, mask=mask)
        conv_sum += x * w_val
    
    # Add bias
    bias_val = tl.load(bias_ptr)
    conv_sum += bias_val
    
    # Store output (this is the pre-softmax logit)
    # The softmax normalization would need a separate reduction
    tl.store(output_ptr + pid, conv_sum, mask=mask)


# Final kernel with actual softmax
@triton.jit
def fused_conv_softmax_final_kernel(
    input_ptr, weight_ptr, bias_ptr, max_buf_ptr, output_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    HW: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Final fused kernel with proper softmax normalization.
    
    Two-phase approach:
    Phase 1: Compute conv sums and find max for numerical stability
    Phase 2: Normalize by exp(x - max) / sum(exp(x - max))
    
    For simplicity, this does one-phase where each program computes
    its softmax value using the pre-computed max in max_buf.
    """
    pid = tl.program_id(0)
    mask = pid < n_elements
    
    # Calculate (b, hw) from program id
    b = pid // HW
    hw = pid % HW
    h = hw // W
    w = hw % W
    
    # Compute conv sum
    base_offset = b * C * H * W + h * W + w
    conv_sum = float(0.0)
    
    for c_idx in range(C):
        x = tl.load(input_ptr + base_offset + c_idx * H * W, mask=mask)
        w_val = tl.load(weight_ptr + c_idx, mask=mask)
        conv_sum += x * w_val
    
    bias_val = tl.load(bias_ptr)
    conv_sum += bias_val
    
    # Get max for this batch (computed in a prior pass)
    batch_max = tl.load(max_buf_ptr + b)
    
    # Compute softmax
    exp_val = tl.exp(conv_sum - batch_max)
    
    # Store output (shape [B, 1, HW, 1])
    tl.store(output_ptr + pid, exp_val, mask=mask)


@torch.fx.wrap
def fused_conv_softmax_kernel_wrapper(in_0, in_1, in_2, B, C, H, W, HW):
    """
    Wrapper for the fused conv + softmax kernel.
    
    Args:
        in_0: bias tensor [1]
        in_1: weight tensor [1, C, 1, 1] 
        in_2: input tensor [B, C, H, W]
        B, C, H, W: dimensions
        
    Returns:
        output tensor [B, 1, HW, 1]
    """
    # Reshape weight to [C]
    weight_1d = in_1.reshape(C)
    
    n_elements = B * HW
    BLOCK_SIZE = min(1024, C)  # Use C if smaller, but at least 1
    
    # Allocate intermediate buffer for conv sums
    conv_buf = torch.empty((B, HW), dtype=in_2.dtype, device=in_2.device)
    max_buf = torch.empty((B,), dtype=in_2.dtype, device=in_2.device)
    
    # Phase 1: Compute conv sums
    grid = (n_elements,)
    
    # First kernel to compute conv sums
    fused_conv_softmax_single_kernel[grid](
        in_2, weight_1d, in_0, conv_buf,
        B, C, H, W, HW,
        n_elements, BLOCK_SIZE
    )
    
    # Compute max per batch for numerical stability
    conv_buf_reshaped = conv_buf.reshape(B, HW)
    for b in range(B):
        max_buf[b] = conv_buf_reshaped[b].max()
    
    # Phase 2: Compute softmax (need to divide by sum of exp across HW)
    # For now, just return the exp(conv - max) 
    # The normalization by sum would be done in a follow-up
    
    output = torch.empty((B, 1, HW, 1), dtype=in_2.dtype, device=in_2.device)
    
    # Final kernel with softmax
    BLOCK_SIZE_final = min(1024, C)
    fused_conv_softmax_final_kernel[grid](
        in_2, weight_1d, in_0, max_buf, output,
        B, C, H, W, HW,
        n_elements, BLOCK_SIZE_final
    )
    
    # Normalize by sum of exp across HW for each batch
    for b in range(B):
        batch_sum = 0.0
        for hw in range(HW):
            batch_sum += output[b, 0, hw, 0]
        for hw in range(HW):
            output[b, 0, hw, 0] = output[b, 0, hw, 0] / (batch_sum + 1e-10)
    
    return output


def pattern(in_0, in_1, in_2):
    """Match the conv + view + softmax + unsqueeze pattern."""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel=5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return tmp_5


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the replacement function."""
    return (in_0, in_1, in_2)


def replacement_func():
    """Return the replacement kernel wrapper."""
    return fused_conv_softmax_kernel_wrapper