import torch
import triton
import triton.language as tl


# Define autotune configurations for better performance
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=3, num_warps=4),
    ],
    key=['n_batch', 'n_channel', 'n_height', 'n_width'],
)
@triton.jit
def fused_bn_relu_kernel(
    input_ptr, add_ptr,  # input tensors
    mean_ptr, var_ptr, weight_ptr, bias_ptr,  # BN parameters
    output_ptr,  # output
    batch_stride, channel_stride, height_stride, width_stride,  # strides for input
    mean_stride, var_stride, weight_stride, bias_stride,  # strides for params
    out_batch_stride, out_channel_stride, out_height_stride, out_width_stride,  # strides for output
    n_batch, n_channel, n_height, n_width,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for: (input + add) -> BatchNorm -> ReLU"""
    
    # Get program ID
    pid = tl.program_id(0)
    num_pid = tl.num_programs(0)
    
    # Calculate elements per program
    num_elements = n_batch * n_channel * n_height * n_width
    elements_per_pid = num_elements // num_pid
    remainder = num_elements % num_pid
    
    # Calculate starting offset for this program
    start_pid = pid * elements_per_pid + tl.minimum(pid, remainder)
    offset = start_pid + tl.arange(0, BLOCK_SIZE)
    
    # Calculate 4D indices
    batch_idx = offset // (n_channel * n_height * n_width)
    rem = offset % (n_channel * n_height * n_width)
    channel_idx = rem // (n_height * n_width)
    rem = rem % (n_height * n_width)
    height_idx = rem // n_width
    width_idx = rem % n_width
    
    # Create masks
    mask = offset < num_elements
    
    # Load input and add value
    input_offset = batch_idx * batch_stride + channel_idx * channel_stride + height_idx * height_stride + width_idx * width_stride
    add_offset = batch_idx * batch_stride + channel_idx * channel_stride + height_idx * height_stride + width_idx * width_stride
    
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    add_val = tl.load(add_ptr + add_offset, mask=mask, other=0.0)
    
    # Element-wise addition
    x = input_val + add_val
    
    # Load BN parameters (per channel)
    mean = tl.load(mean_ptr + channel_idx * mean_stride)
    var = tl.load(var_ptr + channel_idx * var_stride)
    weight = tl.load(weight_ptr + channel_idx * weight_stride)
    bias = tl.load(bias_ptr + channel_idx * bias_stride)
    
    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) * tl.rsqrt(var + eps) * weight + bias
    
    # ReLU: max(0, x)
    result = tl.where(normalized > 0, normalized, 0.0)
    
    # Store result
    out_offset = batch_idx * out_batch_stride + channel_idx * out_channel_stride + height_idx * out_height_stride + width_idx * out_width_stride
    tl.store(output_ptr + out_offset, result, mask=mask)


def fused_batch_norm_add_relu(input_tensor, add_tensor, mean, var, weight, bias, eps=1e-05):
    """Fused kernel wrapper for: (input + add) -> BatchNorm -> ReLU"""
    
    batch, channel, height, width = input_tensor.shape
    output = torch.empty_like(input_tensor)
    
    input_stride = input_tensor.stride()
    output_stride = output.stride()
    
    num_elements = batch * channel * height * width
    BLOCK_SIZE = 1024
    num_programs = min((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, 65536)
    
    fused_bn_relu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        add_ptr=add_tensor,
        mean_ptr=mean,
        var_ptr=var,
        weight_ptr=weight,
        bias_ptr=bias,
        output_ptr=output,
        batch_stride=input_stride[0],
        channel_stride=input_stride[1],
        height_stride=input_stride[2],
        width_stride=input_stride[3],
        mean_stride=mean.stride(0) if mean.dim() > 0 else 0,
        var_stride=var.stride(0) if var.dim() > 0 else 0,
        weight_stride=weight.stride(0) if weight.dim() > 0 else 0,
        bias_stride=bias.stride(0) if bias.dim() > 0 else 0,
        out_batch_stride=output_stride[0],
        out_channel_stride=output_stride[1],
        out_height_stride=output_stride[2],
        out_width_stride=output_stride[3],
        n_batch=batch,
        n_channel=channel,
        n_height=height,
        n_width=width,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3, in_4, in_5):
    """Wrapper function for the fused kernel"""
    output = fused_batch_norm_add_relu(in_5, in_4, in_0, in_1, in_3, in_2, eps=1e-05)
    return in_5 + in_4, output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """Match the exact computation: in_5 + in_4 -> batch_norm -> relu"""
    added = in_5 + in_4
    bn_out = torch.nn.functional.batch_norm(added, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    relu_out = torch.nn.functional.relu(bn_out, inplace=True)
    return added, relu_out


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return kernel_wrapper