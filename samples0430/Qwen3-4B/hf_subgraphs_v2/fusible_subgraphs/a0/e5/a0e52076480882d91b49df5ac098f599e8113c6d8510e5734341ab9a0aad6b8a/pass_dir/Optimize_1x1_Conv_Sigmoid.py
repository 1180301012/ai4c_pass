import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp3 = conv2d.sigmoid()
    tmp4 = in_2 * tmp3
    tmp5 = torch.nn.functional.gelu(tmp4, approximate='none')
    tmp6 = torch.nn.functional.adaptive_avg_pool2d(tmp5, 1)
    tmp7 = tmp6.flatten(1, -1)
    tmp8 = torch.nn.functional.dropout(tmp7, 0.0, False, False)
    return tmp8
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def optimized_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    grid_size: tl.constexpr,
):
    # Each program handles a block of work
    offsets = tl.arange(0, grid_size)
    # Load weights
    weights = tl.load(in_1_ptr + offsets)
    # Load inputs
    input_vals = tl.load(in_3_ptr + offsets)
    # Apply conv (simplified 1x1)
    conv_results = weights * input_vals + tl.load(in_0_ptr + offsets)
    # Sigmoid and other ops
    sigmoid_vals = tl.sigmoid(conv_results)
    multiplied = tl.load(in_2_ptr + offsets) * sigmoid_vals
    gelu_vals = tl.gelu(multiplied, approximate=False)
    # Simplified pooling and flatten
    pooled = tl.mean(gelu_vals)
    tl.store(out_ptr + offsets, pooled)

@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    batch_size = in_0.shape[0]
    in_channels = in_3.shape[1]
    out_channels = in_1.shape[0]
    out = torch.empty((batch_size, out_channels), device=in_0.device, dtype=in_0.dtype)
    grid = (batch_size,)
    optimized_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        in_3_ptr=in_3,
        out_ptr=out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        grid_size=256
    )
    return out
def replacement_func():
    return kernel_wrapper