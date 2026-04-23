import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_add_relu_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    M,
    N,
    stride_x,
    stride_res,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for: linear(x, weight, bias) + residual, followed by relu.
    
    Computes: out = relu(x @ W^T + b + residual)
    
    M = batch size (1000)
    N = feature dimension (128)
    """
    # Get row index for this program
    row_idx = tl.program_id(0)
    
    # Compute offset for this row
    row_offset = row_idx * stride_x
    res_row_offset = row_idx * stride_res
    
    # Initialize accumulator for the linear result
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    # Column indices
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < N
    
    # Loop over the weight matrix K dimension (K=N)
    # Weight shape: [N, N], accessed as W[k, j] = weight_ptr[k * N + j]
    for k in range(0, N, BLOCK_SIZE):
        k_offsets = k + tl.arange(0, BLOCK_SIZE)
        k_mask = k_offsets < N
        
        # Load weight chunk: W[k:k+BLOCK_SIZE, :] shape [BLOCK_SIZE, BLOCK_SIZE]
        w_chunk = tl.load(
            weight_ptr + k_offsets[:, None] * N + col_offsets[None, :],
            mask=k_mask[:, None] & mask[None, :],
            other=0.0
        )
        
        # Load corresponding x values from input row
        x_chunk = tl.load(
            x_ptr + row_offset + k_offsets,
            mask=k_mask,
            other=0.0
        )
        
        # x_chunk is [BLOCK_SIZE], w_chunk is [BLOCK_SIZE, BLOCK_SIZE]
        # x_chunk[:, None] * w_chunk gives [BLOCK_SIZE, BLOCK_SIZE]
        # Sum across K dimension gives [BLOCK_SIZE] - the contribution to each output element
        acc += tl.sum(x_chunk[:, None] * w_chunk, axis=0)
    
    # Load bias and add
    bias = tl.load(bias_ptr + col_offsets, mask=mask, other=0.0)
    result = (acc + bias).to(tl.float16)
    
    # Load residual and add
    residual = tl.load(residual_ptr + res_row_offset + col_offsets, mask=mask, other=0.0)
    result = result + residual
    
    # Apply ReLU
    result = tl.where(result > 0, result, 0.0)
    
    # Store output
    tl.store(out_ptr + res_row_offset + col_offsets, result, mask=mask)


@torch.fx.wrap
def fused_linear_add_relu_dispatch(x, weight, bias, residual, out):
    """
    Dispatch the fused linear + add + relu kernel.
    
    Args:
        x: Input tensor [M, N] on CUDA
        weight: Weight tensor [N, N] on CPU, needs to be moved to CUDA
        bias: Bias tensor [N] on CPU, needs to be moved to CUDA
        residual: Residual tensor [M, N] on CUDA
        out: Output tensor [M, N] on CUDA
    """
    M, N = x.shape
    
    # Move weight and bias to CUDA if needed
    if weight.device.type == 'cpu':
        weight = weight.cuda()
    if bias.device.type == 'cpu':
        bias = bias.cuda()
    
    # Use kernel with BLOCK_SIZE=128
    grid = (M,)
    BLOCK_SIZE = 128
    
    fused_linear_add_relu_kernel[grid](
        x, weight, bias, residual, out,
        M, N,
        x.stride(0), residual.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: linear(in_3, in_1, in_0) + in_2, then relu_
    
    Returns the intermediate result before relu (tmp_3) and final result (tmp_4).
    Since relu_ is in-place on tmp_3, we return tmp_4 which is the relu result.
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    """
    Extract arguments for the fused kernel.
    
    in_0: bias [128]
    in_1: weight [128, 128]
    in_2: residual [1000, 128]
    in_3: input [1000, 128]
    """
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """
    Returns the fused kernel function.
    """
    def fused_kernel(bias, weight, residual, x):
        # Allocate output tensor
        out = torch.empty_like(residual)
        
        # Ensure inputs are on the right device
        x_dev = x.cuda() if x.device.type == 'cpu' else x
        weight_dev = weight.cuda() if weight.device.type == 'cpu' else weight
        bias_dev = bias.cuda() if bias.device.type == 'cpu' else bias
        
        return fused_linear_add_relu_dispatch(x_dev, weight_dev, bias_dev, residual, out)
    
    return fused_kernel