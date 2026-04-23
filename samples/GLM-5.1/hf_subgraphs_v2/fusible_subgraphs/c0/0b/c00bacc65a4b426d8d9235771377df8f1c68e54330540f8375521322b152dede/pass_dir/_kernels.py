import torch
import triton
import triton.language as tl


@triton.jit
def bn_kernel(
    input_ptr, running_mean_ptr, running_var_ptr, weight_ptr, bias_ptr, output_ptr,
    C, HW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Compute channel index for each element
    # For [N, C, H, W]: offset = n*C*H*W + c*H*W + h*W + w
    # channel = (offset // HW) % C  where HW = H*W
    channel_offsets = (offsets // HW) % C

    eps = 1e-05
    # Load BN params and compute in float32 for numerical stability
    running_mean_val = tl.load(running_mean_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)
    running_var_val = tl.load(running_var_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
    weight_val = tl.load(weight_ptr + channel_offsets, mask=mask, other=1.0).to(tl.float32)
    bias_val = tl.load(bias_ptr + channel_offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute scale and shift per-channel
    inv_std = 1.0 / tl.sqrt(running_var_val + eps)
    scale = weight_val * inv_std
    shift = bias_val - running_mean_val * scale

    # Load input and apply BN transform
    input_val = tl.load(input_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    output_val = input_val * scale + shift

    # Store output
    tl.store(output_ptr + offsets, output_val, mask=mask)


@triton.jit
def avg_pool2d_kernel(
    input_ptr, output_ptr,
    N, C, H_in, W_in, H_out, W_out,
    total_out_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_out_elements

    # Decode output coordinates
    ow = offsets % W_out
    oh = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    n = offsets // (W_out * H_out * C)

    # Input coordinates: ih = oh*2, iw = ow*2
    ih = oh * 2
    iw = ow * 2

    # Base input offset for (n, c)
    base = n * C * H_in * W_in + c * H_in * W_in

    # Load 4 input values for 2x2 pooling
    v00 = tl.load(input_ptr + base + ih * W_in + iw, mask=mask, other=0.0)
    v01 = tl.load(input_ptr + base + ih * W_in + iw + 1, mask=mask, other=0.0)
    v10 = tl.load(input_ptr + base + (ih + 1) * W_in + iw, mask=mask, other=0.0)
    v11 = tl.load(input_ptr + base + (ih + 1) * W_in + iw + 1, mask=mask, other=0.0)

    avg = (v00 + v01 + v10 + v11) * 0.25

    tl.store(output_ptr + offsets, avg, mask=mask)


@torch.fx.wrap
def triton_batch_norm(input, running_mean, running_var, weight, bias):
    # Ensure all tensors are on the same device and contiguous
    device = input.device
    running_mean = running_mean.to(device=device).contiguous()
    running_var = running_var.to(device=device).contiguous()
    weight = weight.to(device=device).contiguous()
    bias = bias.to(device=device).contiguous()
    input = input.contiguous()

    N = input.shape[0]
    C = input.shape[1]
    H = input.shape[2]
    W = input.shape[3]
    HW = H * W
    total_elements = N * C * H * W

    output = torch.empty_like(input)

    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    bn_kernel[(num_programs,)](
        input_ptr=input, running_mean_ptr=running_mean, running_var_ptr=running_var,
        weight_ptr=weight, bias_ptr=bias, output_ptr=output,
        C=C, HW=HW, total_elements=total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@torch.fx.wrap
def triton_avg_pool2d(input):
    input = input.contiguous()

    N = input.shape[0]
    C = input.shape[1]
    H_in = input.shape[2]
    W_in = input.shape[3]
    H_out = H_in // 2
    W_out = W_in // 2
    total_out_elements = N * C * H_out * W_out

    output = torch.empty((N, C, H_out, W_out), dtype=input.dtype, device=input.device)

    BLOCK_SIZE = 1024
    num_programs = (total_out_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    avg_pool2d_kernel[(num_programs,)](
        input_ptr=input, output_ptr=output,
        N=N, C=C, H_in=H_in, W_in=W_in, H_out=H_out, W_out=W_out,
        total_out_elements=total_out_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output


@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "bn":
        return triton_batch_norm(args[0], args[1], args[2], args[3], args[4])
    elif route == "avgpool":
        return triton_avg_pool2d(args[0])
    else:
        raise ValueError(f"Unknown route: {route}")