import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_5[slice(None, None, None), slice(64, None, None), slice(None, None, None), slice(None, None, None)]
    tmp_5 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)

@triton.jit
def batch_norm_inference_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = N * C * H * W
    mask = offsets < total_elements
    
    c_indices = (offsets // (H * W)) % C
    
    mean = tl.load(mean_ptr + c_indices, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_indices, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c_indices, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_indices, mask=mask, other=0.0)
    
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * weight + bias
    
    tl.store(output_ptr + offsets, x_norm, mask=mask)

@torch.fx.wrap
def optimized_batch_norm_and_slice(in_0, in_1, in_2, in_3, in_4, in_5):
    N, C, H, W = in_4.shape
    output_bn = torch.empty_like(in_4)
    
    eps = 0.001
    BLOCK_SIZE = 1024
    total_elements = N * C * H * W
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    batch_norm_inference_kernel[grid](
        in_4, output_bn,
        in_0, in_1, in_3, in_2,
        N, C, H, W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    slice_output = in_5[:, 64:, :, :]
    
    return (output_bn, slice_output)

def replacement_func():
    return optimized_batch_norm_and_slice