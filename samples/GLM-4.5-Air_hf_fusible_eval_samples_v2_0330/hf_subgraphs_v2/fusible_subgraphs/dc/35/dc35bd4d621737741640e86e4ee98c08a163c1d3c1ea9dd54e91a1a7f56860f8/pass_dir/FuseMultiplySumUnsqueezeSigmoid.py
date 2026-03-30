import torch
import triton
import triton.language as tl

def pattern(x, y):
    """
    Pattern matching for the computation:
    tmp_0 = x * y
    tmp_1 = torch.sum(tmp_0, dim=1)  
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3
    """
    tmp_0 = x * y
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

def replacement_args(x, y):
    return (x, y)

@triton.jit
def fused_elementwise_kernel(
    x_ptr,
    y_ptr,
    mul_ptr,
    B: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
):
    """Element-wise multiplication kernel"""
    pid = tl.program_id(0)
    grid_size = B * H * W
    
    if pid >= grid_size:
        return
    
    # Calculate spatial position (excluding batch and channel)
    spatial_idx = pid
    b = spatial_idx // (H * W)
    h = (spatial_idx % (H * W)) // W
    w = spatial_idx % W
    
    # Load and multiply all channels for this spatial position
    sum_result = 0.0
    for c in range(C):
        x_offset = (b * C + c) * H * W + h * W + w
        y_offset = (b * C + c) * H * W + h * W + w
        
        x_val = tl.load(x_ptr + x_offset)
        y_val = tl.load(y_ptr + y_offset)
        sum_result += x_val * y_val
    
    # Store the sum result (after reduction along channels)
    output_offset = b * H * W + h * W + w
    tl.store(mul_ptr + output_offset, sum_result)

@triton.jit
def fused_sigmoid_kernel(
    sigmoid_input_ptr,
    out_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    input_dtype: tl.constexpr,
):
    """Sigmoid activation kernel with shape expansion [B, H, W] -> [B, 1, H, W]"""
    pid = tl.program_id(0)
    grid_size = B * H * W
    
    if pid >= grid_size:
        return
    
    # Calculate spatial position
    spatial_idx = pid
    b = spatial_idx // (H * W)
    h = (spatial_idx % (H * W)) // W
    w = spatial_idx % W
    
    # Load reduced value and convert to fp32 for sigmoid computation
    input_offset = b * H * W + h * W + w
    val = tl.load(sigmoid_input_ptr + input_offset)
    val_fp32 = val.to(tl.float32)
    
    # Apply sigmoid using fp32 precision
    sigmoid_fp32 = 1.0 / (1.0 + tl.exp(-val_fp32))
    
    # Convert back to original data type
    if input_dtype == "bf16":
        sigmoid_val = sigmoid_fp32.to(tl.bfloat16)
    elif input_dtype == "float16":
        sigmoid_val = sigmoid_fp32.to(tl.float16)
    elif input_dtype == "float32":
        sigmoid_val = sigmoid_fp32  # Already float32
    else:
        # Default to float32 for unknown types
        sigmoid_val = sigmoid_fp32
    
    # Store result with expanded dimensions (B, 1, H, W)
    # Each spatial position corresponds to position (b, 0, h, w) in output
    output_offset = (b * 1 + 0) * H * W + h * W + w
    tl.store(out_ptr + output_offset, sigmoid_val)

@torch.fx.wrap
def fused_multiply_sum_sigmoid(x, y):
    """Fused operation: multiply + sum + unsqueeze + sigmoid"""
    
    # Get input tensor properties
    if x.dim() != 4 or y.dim() != 4:
        raise ValueError("Input tensors must be 4D")
    
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    
    B, C, H, W = x.shape
    
    # Step 1: Perform element-wise multiplication and reduce sum along channels (dim=1)
    # Output will be [B, H, W] after reduction
    intermediate_shape = (B, H, W)
    intermediate = torch.empty(intermediate_shape, dtype=x.dtype, device=x.device)
    
    # Launch element-wise + reduction kernel
    fused_elementwise_kernel[(B * H * W,)](
        x_ptr=x,
        y_ptr=y,
        mul_ptr=intermediate,
        B=B,
        C=C,
        H=H,
        W=W,
    )
    
    # Step 2: Apply sigmoid with unsqueeze expansion [B, H, W] -> [B, 1, H, W]
    output_shape = (B, 1, H, W)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Map torch dtype to Triton dtype string
    dtype_str = str(x.dtype).replace('torch.', '')
    if x.dtype == torch.bfloat16:
        dtype_str = "bf16"
    elif x.dtype == torch.float16:
        dtype_str = "float16"
    elif x.dtype == torch.float32:
        dtype_str = "float32"
    
    fused_sigmoid_kernel[(B * H * W,)](
        sigmoid_input_ptr=intermediate,
        out_ptr=out,
        B=B,
        H=H,
        W=W,
        input_dtype=dtype_str,
    )
    
    return out

def replacement_func():
    return fused_multiply_sum_sigmoid