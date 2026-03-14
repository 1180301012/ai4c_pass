import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def fused_add_layernorm_kernel(
    input1_ptr, input2_ptr, weight_ptr, bias_ptr, output_ptr,
    N, B, H, W,
    stride_input1, stride_input2, stride_weight, stride_bias, stride_output,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    num_elements = B * H * W
    
    if program_id >= num_elements:
        return
    
    base_offset = program_id * N
    offs = tl.arange(0, BLOCK_SIZE)[:N]
    
    weight = tl.load(weight_ptr + offs)
    bias = tl.load(bias_ptr + offs)
    
    input1 = tl.load(input1_ptr + base_offset + offs)
    input2 = tl.load(input2_ptr + base_offset + offs)
    add_result = input1 + input2
    
    mean = tl.sum(add_result, axis=0) / N
    var = tl.sum((add_result - mean) * (add_result - mean), axis=0) / N
    std = tl.sqrt(var + eps)
    
    normalized = (add_result - mean) / std
    output = normalized * weight + bias
    
    tl.store(output_ptr + base_offset + offs, output)


@torch.fx.wrap
def fused_add_layernorm(x1, x2, weight, bias, normalized_shape, eps=1e-05):
    """Fused add + layer_norm kernel for batch 32 face-parsing variant."""
    B, HW, N = x1.shape
    H = W = 16
    
    output = torch.empty_like(x1)
    
    num_elements = B * H * W
    grid = (num_elements,)
    
    fused_add_layernorm_kernel[grid](
        x1, x2, weight, bias, output,
        N, B, H, W,
        x1.stride(0), x2.stride(0), weight.stride(0), bias.stride(0), output.stride(0),
        eps,
        BLOCK_SIZE=1024,
    )
    
    return output


def pattern(in_0, in_1, in_2, in_3, in_4):
    """Match add + layer_norm pattern for face-parsing batch 32."""
    tmp_2 = in_4 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (512,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(32, 16, 16, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = in_2.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    return tmp_6, tmp_8


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_3, in_4, in_1, in_0)


def replacement_func():
    return fused_add_layernorm