import torch
import triton
import triton.language as tl

def pattern(bn_out):
    # BatchNorm + ReLU fusion pattern
    relu_out = torch.nn.functional.relu(bn_out, inplace=False)
    return bn_out, relu_out

def replacement_args(bn_out):
    return (bn_out,)

@triton.jit
def fused_bn_relu_kernel(
    bn_out_ptr, relu_out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    num_programs = tl.cdiv(N * C * H * W, BLOCK_SIZE)
    
    if pid >= num_programs:
        return
    
    # Each program handles BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N * C * H * W)
    
    # Load batch norm output
    bn_values = tl.load(bn_out_ptr + offsets, mask=mask, other=0.0)
    
    # ReLU activation
    relu_values = tl.maximum(bn_values, 0.0)
    
    # Store result
    tl.store(relu_out_ptr + offsets, relu_values, mask=mask)

@torch.fx.wrap
def fused_bn_relu(bn_out):
    N, C, H, W = bn_out.shape
    
    # Output tensor
    relu_out = torch.empty_like(bn_out)
    
    # Optimize block size based on tensor size
    total_elements = N * C * H * W
    if total_elements < 1024:
        BLOCK_SIZE = 128
    elif total_elements < 10000:
        BLOCK_SIZE = 256
    elif total_elements < 100000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_bn_relu_kernel[(num_programs,)](
        bn_out_ptr=bn_out, relu_out_ptr=relu_out,
        N=N, C=C, H=H, W=W,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return relu_out

def replacement_func():
    return fused_bn_relu