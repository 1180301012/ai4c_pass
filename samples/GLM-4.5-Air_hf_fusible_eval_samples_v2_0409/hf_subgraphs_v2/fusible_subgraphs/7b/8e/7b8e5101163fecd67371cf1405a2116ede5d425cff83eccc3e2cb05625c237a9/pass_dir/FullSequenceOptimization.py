import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = in_1 * tmp_2
    tmp_4 = tmp_3 + in_0
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def full_fusion_kernel(
    bias_ptr,
    scale_ptr,
    activation_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load bias, scale, and activation
    bias = tl.load(bias_ptr, element_type=tl.float16)
    scale = tl.load(scale_ptr, element_type=tl.float16)
    x = tl.load(activation_ptr + offsets, mask=mask, other=0.0)
    
    # Fused computation: ReLU -> Scale -> Add
    # For bfloat16/float16, we need to handle precision carefully
    relu_result = tl.where(x > 0, x, 0.0)
    scaled_result = relu_result * scale
    final_result = scaled_result + bias
    
    # Store result
    tl.store(out_ptr + offsets, final_result, mask=mask)

@torch.fx.wrap
def fused_bias_activation_scale_relu(bias, scale, activation):
    # Choose optimal block size based on tensor size
    if activation.numel() >= 1024 * 1024:  # Large tensors
        BLOCK_SIZE = 2048
    elif activation.numel() >= 256 * 1024:  # Medium tensors
        BLOCK_SIZE = 1024
    else:  # Small tensors
        BLOCK_SIZE = 512
    
    N = activation.numel()
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(activation)
    
    full_fusion_kernel[(num_programs,)](
        bias_ptr=bias,
        scale_ptr=scale,
        activation_ptr=activation,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_bias_activation_scale_relu