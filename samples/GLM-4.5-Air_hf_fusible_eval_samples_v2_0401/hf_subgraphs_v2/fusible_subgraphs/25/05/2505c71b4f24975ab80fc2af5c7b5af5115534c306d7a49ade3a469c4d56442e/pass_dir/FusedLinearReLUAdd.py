import torch
import triton
import triton.language as tl

def pattern(x, scale, bias):
    # This matches the exact computation: ReLU -> Mul -> Add sequence
    relu_out = torch.nn.functional.relu(x, inplace=False)
    mul_out = scale * relu_out
    add_out = mul_out + bias
    # Return only the final result since it's what gets used later
    return add_out

def replacement_args(x, scale, bias):
    return (x, scale, bias)

@triton.jit
def fused_linear_relu_add_kernel(
    x_ptr,
    scale_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a block of the spatial dimensions
    pid = tl.program_id(0)
    total_elements = N * C * H * W
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Reshape offsets to 4D for 2D operations
    w = offsets % W
    h = (offsets // W) % H
    c = (offsets // (W * H)) % C
    n = offsets // (W * H * C)
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + [0], mask=mask, other=1.0)  # scale is [1]
    bias = tl.load(bias_ptr + [0], mask=mask, other=0.0)    # bias is [1]
    
    # Fused operations: relu(x) * scale + bias
    relu_x = tl.maximum(x, 0.0)
    out = relu_x * scale + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_linear_relu_add(x, scale, bias):
    N, C, H, W = x.shape
    total_elements = N * C * H * W
    BLOCK_SIZE = 1024  # Optimal block size for GPU
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x, dtype=x.dtype, device=x.device)
    
    fused_linear_relu_add_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        bias_ptr=bias,
        out_ptr=out,
        N=N,
        C=C,
        H=H,
        W=W,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_linear_relu_add