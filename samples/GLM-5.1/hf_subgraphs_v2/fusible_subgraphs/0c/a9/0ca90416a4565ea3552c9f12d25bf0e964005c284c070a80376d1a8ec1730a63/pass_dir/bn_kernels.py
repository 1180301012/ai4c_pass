import torch
import triton
import triton.language as tl


@triton.jit
def bn_inference_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    eps: tl.constexpr,
    n_elements, C, HW,
    BLOCK_SIZE: tl.constexpr,
    DTYPE_OUT: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Compute channel index for each element
    # offset in flat [N, C, H, W] = n * C * HW + c * HW + hw
    # channel = (offset // HW) % C
    channel = (offsets // HW) % C

    # Load BN params and promote to float32 for numerical accuracy
    mean_val = tl.load(mean_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    var_val = tl.load(var_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    w_val = tl.load(weight_ptr + channel, mask=mask, other=0.0).to(tl.float32)
    b_val = tl.load(bias_ptr + channel, mask=mask, other=0.0).to(tl.float32)

    # Compute scale and offset: output = input * scale + offset
    # scale = weight / sqrt(var + eps)
    # offset = bias - mean * scale
    inv_std = 1.0 / tl.sqrt(var_val + eps)
    scale = w_val * inv_std
    offset_val = b_val - mean_val * scale

    # Load input, promote to float32, compute affine transform
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = (x * scale + offset_val).to(DTYPE_OUT)
    tl.store(output_ptr + offsets, out, mask=mask)


def _get_triton_dtype(torch_dtype):
    if torch_dtype == torch.float32:
        return tl.float32
    elif torch_dtype == torch.float16:
        return tl.float16
    elif torch_dtype == torch.bfloat16:
        return tl.bfloat16
    else:
        return tl.float32


@torch.fx.wrap
def bn_dispatch(running_mean, running_var, bias, weight, input_tensor):
    # Move BN params to input device if needed (they may be on CPU)
    # .to() uses aten._to_copy which is in the PosionDispatchTensor whitelist
    target_device = input_tensor.device
    running_mean = running_mean.to(target_device)
    running_var = running_var.to(target_device)
    bias = bias.to(target_device)
    weight = weight.to(target_device)

    # Get shape info
    N, C, H, W = input_tensor.shape
    HW = H * W
    n_elements = N * C * H * W

    # Allocate output
    output = torch.empty_like(input_tensor)

    # Determine Triton dtype for output casting
    DTYPE_OUT = _get_triton_dtype(input_tensor.dtype)

    # Choose block size based on tensor size for better parallelism
    if n_elements < 100000:
        BLOCK_SIZE = 256
    elif n_elements < 500000:
        BLOCK_SIZE = 512
    else:
        BLOCK_SIZE = 1024

    # Grid: 1D flat over all elements
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch Triton kernel
    bn_inference_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        mean_ptr=running_mean,
        var_ptr=running_var,
        weight_ptr=weight,
        bias_ptr=bias,
        eps=0.001,
        n_elements=n_elements,
        C=C,
        HW=HW,
        BLOCK_SIZE=BLOCK_SIZE,
        DTYPE_OUT=DTYPE_OUT,
    )

    return output