import torch
import triton
import triton.language as tl

def pattern(conv_output, mul_input):
    # Match the sigmoid -> multiplication pattern exactly as in the model
    tmp_3 = torch.sigmoid(conv_output)
    tmp_4 = mul_input * tmp_3
    return tmp_3, tmp_4

def replacement_args(conv_output, mul_input):
    return (conv_output, mul_input)

@triton.jit
def fused_sigmoid_mul_kernel(
    conv_out_ptr,
    mul_in_ptr,
    sigmoid_out_ptr,
    mul_out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    conv_out = tl.load(conv_out_ptr + offsets, mask=mask, other=0.0)
    mul_in = tl.load(mul_in_ptr + offsets, mask=mask, other=0.0)
    
    # Fused sigmoid + multiplication
    sigmoid_val = 1.0 / (1.0 + tl.exp(-conv_out))
    mul_val = mul_in * sigmoid_val
    
    # Store both results
    tl.store(sigmoid_out_ptr + offsets, sigmoid_val, mask=mask)
    tl.store(mul_out_ptr + offsets, mul_val, mask=mask)

@torch.fx.wrap
def fused_sigmoid_mul(conv_output, mul_input):
    # Use the Triton kernel for fusion
    out1, out2 = fused_sigmoid_mul_torch(conv_output, mul_input)
    return out1, out2

@torch.fx.wrap 
def fused_sigmoid_mul_torch(conv_output, mul_input):
    # Get shapes and create output tensors
    conv_shape = conv_output.shape
    mul_shape = mul_input.shape
    
    # Determine total elements for processing
    if len(mul_shape) == 4:  # [B, C, H, W] - expand conv_output to match
        total_elements = mul_input.numel()
        sigmoid_out = torch.empty(mul_shape, dtype=conv_output.dtype, device=conv_output.device)
        mul_out = torch.empty_like(mul_input)
    else:  # Use original shapes
        total_elements = conv_output.numel()
        sigmoid_out = torch.empty_like(conv_output)
        mul_out = torch.empty_like(mul_input)
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_sigmoid_mul_kernel[(num_programs,)](
        conv_out_ptr=conv_output,
        mul_in_ptr=mul_input,
        sigmoid_out_ptr=sigmoid_out,
        mul_out_ptr=mul_out, 
        n_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return sigmoid_out, mul_out

def replacement_func():
    return fused_sigmoid_mul