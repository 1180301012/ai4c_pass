import torch
import triton
import triton.language as tl

def pattern(x, weight, bias):
    # pattern matches: reshape + linear operations
    tmp_reshaped = x.reshape(300, 1, 256)  # [1,150,1,512] -> [300,1,256]
    tmp_linear = torch.nn.functional.linear(tmp_reshaped, weight, bias)
    return tmp_linear

def replacement_args(x, weight, bias):
    return (x, weight, bias)

@triton.jit
def reshape_linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_original_dims,
    total_elements,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one output position
    pid = tl.program_id(0)
    batch = pid // (1 * 512 // 256)  # accounting for reshape dims
    inner = (pid % (1 * 512 // 256)) // (512 // 256)
    feat = (pid % (512 // 256))
    
    # Calculate input position after reshape
    # Original: [1,150,1,512] -> reshaped: [300,1,256]
    input_idx = batch * (150 * 1 * 512) + inner * (1 * 512) + (feat * 2)  # feat*2 because we go from 256->512
    
    # Load input and weight
    x_vals = tl.load(x_ptr + input_idx + tl.arange(0, 2), mask=input_idx + tl.arange(0, 2) < total_elements)
    weights = tl.load(weight_ptr + tl.arange(0, 512 * 2).reshape(512, 2))
    
    # Compute linear transformation
    output = tl.sum(x_vals.reshape(1, 2) @ weights, axis=1) + bias[feat]
    
    # Store output
    output_idx = batch * (1 * 512) + inner * 512 + feat
    tl.store(out_ptr + output_idx, output, mask=output_idx < total_elements // 2)

@torch.fx.wrap
def triton_reshape_linear(x, weight, bias):
    total_elements = x.numel()
    BLOCK_M = 1
    BLOCK_K = 256
    
    # Output shape: [300, 1, 512]
    output_shape = (300, 1, 512)
    out = torch.empty(output_shape, dtype=x.dtype, device='cuda')
    
    # Launch kernel
    num_programs = (300 * 1 * 512 + BLOCK_M - 1) // BLOCK_M
    reshape_linear_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_original_dims=4,
        total_elements=total_elements,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K
    )
    
    return out

def replacement_func():
    return triton_reshape_linear