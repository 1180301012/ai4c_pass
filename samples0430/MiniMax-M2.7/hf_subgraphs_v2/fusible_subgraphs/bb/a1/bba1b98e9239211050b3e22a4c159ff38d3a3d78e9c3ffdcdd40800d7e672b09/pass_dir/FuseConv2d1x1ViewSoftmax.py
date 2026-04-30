import torch
import triton
import triton.language as tl

# Pattern matching: Conv2d (1x1) -> View -> Softmax
# Using literal 1 for batch to match the graphs that have batch=1
def pattern(in_0, in_1, in_2):
    """
    Match the pattern: Conv2D (1x1 kernel) -> View -> Softmax
    """
    # Conv2d with 1x1 kernel, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    # View operation to reshape (using literal 1 for batch)
    tmp_3 = conv2d.view(1, 1, -1)
    # Softmax over last dimension
    tmp_4 = tmp_3.softmax(dim=-1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments needed for the replacement function.
    """
    # Return the route string for dispatch
    return (in_0, in_1, in_2)


# Autotune configuration for the fused kernel
@triton.autotune(
    [
        triton.Config({"BLOCK_SIZE": 256}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_stages=3, num_warps=4),
    ],
    key=["n_elements"],
)
@triton.jit
def fused_conv_softmax_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    n_elements,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Conv2d(1x1) + View + Softmax
    
    The kernel performs:
    1. Conv2D 1x1: output[b, 0, h, w] = sum_c(in[b, c, h, w] * weight[0, c, 0, 0]) + bias[0]
    2. View: reshape to [B, 1, H*W]
    3. Softmax: exp(x - max) / sum(exp(x - max)) over last dimension
    
    Args:
        input_ptr: input tensor [B, C, H, W] as flat pointer
        weight_ptr: weight tensor [1, C, 1, 1] - accessed as weight[c] (since stride is [C*1*1, 1*1, 1, 1])
        bias_ptr: bias tensor [1]
        output_ptr: output tensor [B, H*W] (after view)
        n_elements: total elements in output (B * H*W)
        num_channels: C (512)
        height: H (64)
        width: W (64)
    """
    # Get batch index from program id
    pid = tl.program_id(0)
    
    # Calculate output position for this program
    output_offset = pid * BLOCK_SIZE
    offsets = output_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute input base offset for this batch: b * C * H * W
    base_input_offset = pid * num_channels * height * width
    
    # Step 1: Compute Conv2D 1x1 (dot product over channels)
    # For 1x1 conv with weight [1, C, 1, 1], groups=1, output[b, 0, h, w] = sum_c(in[b,c,h,w] * weight[0,c,0,0])
    # Weight shape is [1, C, 1, 1], so weight[0, c, 0, 0] is at offset c (with stride [C*1*1, 1*1, 1, 1])
    conv_result = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    for c in range(num_channels):
        # Load weight for this channel: weight[0, c, 0, 0]
        weight_val = tl.load(weight_ptr + c)
        
        # Compute input offset for all spatial positions in this channel
        spatial_offsets = tl.arange(0, BLOCK_SIZE)
        
        # Compute h, w for each spatial position
        h_vals = spatial_offsets // width
        w_vals = spatial_offsets % width
        
        # Input offset: b * C * H * W + c * H * W + h * W + w
        input_offsets = base_input_offset + c * height * width + h_vals * width + w_vals
        
        # Load input values for this channel
        input_vals = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Accumulate: result += input * weight
        conv_result = conv_result + input_vals * weight_val
    
    # Step 2: Add bias
    bias_val = tl.load(bias_ptr)
    conv_result = conv_result + bias_val
    
    # Step 3: Store intermediate result
    tl.store(output_ptr + offsets, conv_result, mask=mask)


@triton.jit
def softmax_max_kernel(
    input_ptr,
    max_ptr,
    n_elements,
    num_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """
    First pass of softmax: compute max for each batch
    """
    pid = tl.program_id(0)
    
    # Each program handles one batch
    batch_start = pid * num_spatial
    offsets = batch_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * num_spatial
    
    # Load all values and find max
    vals = tl.load(input_ptr + offsets, mask=mask, other=float("-inf"))
    max_val = tl.max(vals, axis=0)
    
    tl.store(max_ptr + pid, max_val)


@triton.jit
def softmax_exp_sum_kernel(
    input_ptr,
    max_ptr,
    exp_sum_ptr,
    output_ptr,
    n_elements,
    num_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Second pass of softmax: compute exp(x - max) and sum for each batch
    """
    pid = tl.program_id(0)
    
    # Load max for this batch
    batch_max = tl.load(max_ptr + pid)
    
    # Each program handles one batch
    batch_start = pid * num_spatial
    offsets = batch_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * num_spatial
    
    # Load values, compute exp(x - max)
    vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    exp_vals = tl.exp(vals - batch_max)
    
    # Store exp values
    tl.store(output_ptr + offsets, exp_vals, mask=mask)
    
    # Compute sum of exp values for this batch
    exp_sum = tl.sum(exp_vals, axis=0)
    tl.store(exp_sum_ptr + pid, exp_sum)


@triton.jit
def softmax_normalize_kernel(
    exp_ptr,
    exp_sum_ptr,
    output_ptr,
    n_elements,
    num_spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Third pass of softmax: normalize by dividing by sum
    """
    pid = tl.program_id(0)
    
    # Load exp sum for this batch
    batch_sum = tl.load(exp_sum_ptr + pid)
    
    # Each program handles one batch
    batch_start = pid * num_spatial
    offsets = batch_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (pid + 1) * num_spatial
    
    # Load exp values and normalize
    exp_vals = tl.load(exp_ptr + offsets, mask=mask, other=0.0)
    normalized = exp_vals / batch_sum
    
    # Store final result (output is [B, 1, H*W], stored as [B*H*W] with stride B*1*HW)
    tl.store(output_ptr + offsets, normalized, mask=mask)


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2):
    """
    Wrapper for the fused Conv2d + View + Softmax kernel.
    
    Uses Triton kernels for the computation.
    """
    B, C, H, W = in_2.shape
    num_spatial = H * W
    n_elements = B * num_spatial
    
    # Allocate output tensor: [B, H*W]
    # Note: We allocate [B, num_spatial] but return with unsqueeze to make [B, 1, H*W]
    output = torch.empty((B, num_spatial), dtype=in_2.dtype, device=in_2.device)
    
    # Grid configuration: one program per batch
    grid = (B,)
    
    # Launch kernel to compute conv result
    fused_conv_softmax_kernel[grid](
        in_2,           # input_ptr - tensor [B, C, H, W]
        in_1,           # weight_ptr - tensor [1, C, 1, 1], accessed as [c]
        in_0,           # bias_ptr
        output,         # output_ptr - [B, H*W]
        n_elements,     # n_elements
        C,              # num_channels
        H,              # height
        W,              # width
    )
    
    # Now apply softmax
    # Allocate intermediate buffers
    max_buffer = torch.zeros(B, dtype=in_2.dtype, device=in_2.device)
    exp_sum_buffer = torch.zeros(B, dtype=in_2.dtype, device=in_2.device)
    exp_buffer = torch.empty_like(output)
    
    # BLOCK_SIZE for softmax kernels - should be >= num_spatial (4096)
    softmax_block_size = 4096
    
    # Step 1: Find max per batch
    softmax_max_kernel[(B,)](
        output, max_buffer, n_elements, num_spatial, softmax_block_size
    )
    
    # Step 2: Compute exp(x - max) and sum
    softmax_exp_sum_kernel[(B,)](
        output, max_buffer, exp_sum_buffer, exp_buffer, n_elements, num_spatial, softmax_block_size
    )
    
    # Step 3: Normalize
    softmax_normalize_kernel[(B,)](
        exp_buffer, exp_sum_buffer, output, n_elements, num_spatial, softmax_block_size
    )
    
    # Return output - reshape to [B, 1, H*W]
    # Since unsqueeze might be blocked, use reshape
    return output.reshape(B, 1, num_spatial)


def replacement_func():
    """
    Returns the replacement function that implements Conv2d + View + Softmax fusion.
    """
    return fused_kernel_wrapper