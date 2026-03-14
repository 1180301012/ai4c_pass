import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """Match element-wise multiplication followed by sum, unsqueeze, and sigmoid"""
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3

def replacement_args(in_0, in_1):
    """Extract input arguments for the replacement"""
    return (in_0, in_1)

@triton.jit
def fused_kernel(
    x_ptr, 
    y_ptr, 
    out_ptr,
    batch_size: tl.constexpr,
    dim1_size: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    """Fused kernel: element-wise multiplication + sum + sigmoid"""
    # Each program handles one position in the output [batch, 1, height, width]
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute batch and spatial coordinates
    batch = pid_m // (height * width)
    h = (pid_m // width) % height
    w = pid_m % width
    
    # Bounds checking
    if batch >= batch_size:
        return
    
    # Compute pointer offsets for this spatial position
    x_base = batch * dim1_size * height * width + h * width + w
    y_base = batch * dim1_size * height * width + h * width + w
    
    # Process a block along the sum dimension (dim=1)
    offsets = tl.arange(0, BLOCK_SIZE_M)
    block_mask = offsets < dim1_size
    
    # Load blocks from input tensors
    x_block = tl.load(x_ptr + x_base + offsets * height * width, mask=block_mask, other=0.0)
    y_block = tl.load(y_ptr + y_base + offsets * height * width, mask=block_mask, other=0.0)
    
    # Element-wise multiplication and sum along dim=1
    product = x_block * y_block
    sum_val = tl.sum(product * block_mask)  # Mask unused elements in sum
    
    # Apply sigmoid
    sigmoid_val = 1.0 / (1.0 + tl.exp(-sum_val))
    
    # Store result at output position [batch, 0, h, w]
    out_idx = batch * height * width + h * width + w
    tl.store(out_ptr + out_idx, sigmoid_val)

@torch.fx.wrap
def fused_computation(in_0, in_1):
    """Wrapper for the fused computation kernel"""
    B, C, H, W = in_0.shape
    out = torch.empty((B, 1, H, W), dtype=in_0.dtype, device=in_0.device)
    
    # Kernel launch configuration
    BLOCK_SIZE_M = 1024  # Block size for sum dimension
    BLOCK_SIZE_N = 1     # Blocks per spatial position
    
    # Grid dimensions: [batch_height_width_batch_size, 1]
    grid_size = (B * H * W, 1)
    
    fused_kernel[grid_size](
        in_0, in_1, out, 
        batch_size=B,
        dim1_size=C,
        height=H,
        width=W,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N
    )
    
    return out

def replacement_func():
    """Return the fused computation function"""
    return fused_computation