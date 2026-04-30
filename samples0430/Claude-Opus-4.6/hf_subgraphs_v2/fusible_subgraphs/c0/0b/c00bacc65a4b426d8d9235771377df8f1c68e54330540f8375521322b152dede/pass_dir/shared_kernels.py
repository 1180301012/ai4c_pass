import torch
import triton
import triton.language as tl


@triton.jit
def bn_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, HW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # 1D grid with large BLOCK_SIZE to minimize program count
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Channel index (NCHW layout): c = (linear_idx // HW) % C
    c_idx = (idx // HW) % C

    # Load input
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # Load per-channel BN params (gather, but L1 cached for adjacent threads)
    mean_val = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var_val = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w_val = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b_val = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # BN inference: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = tl.rsqrt(var_val + 1e-5)
    result = (x_f32 - mean_val) * inv_std * w_val + b_val

    # Store
    tl.store(output_ptr + idx, result.to(x.dtype), mask=mask)


@triton.jit
def pool_kernel(
    input_ptr, output_ptr,
    HW, W, HW_out, W_out,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    hw_out_idx = idx % HW_out
    nc_idx = idx // HW_out

    h_out = hw_out_idx // W_out
    w_out = hw_out_idx % W_out

    h_in = h_out * 2
    w_in = w_out * 2

    input_base = nc_idx * HW

    off_base = input_base + h_in * W + w_in
    val00 = tl.load(input_ptr + off_base, mask=mask, other=0.0)
    val01 = tl.load(input_ptr + off_base + 1, mask=mask, other=0.0)
    val10 = tl.load(input_ptr + off_base + W, mask=mask, other=0.0)
    val11 = tl.load(input_ptr + off_base + W + 1, mask=mask, other=0.0)

    result = (val00 + val01 + val10 + val11) * 0.25

    tl.store(output_ptr + idx, result, mask=mask)


@torch.fx.wrap
def dispatch_wrapper(*args):
    if len(args) == 5:
        # Batch norm inference
        input_tensor = args[0]
        running_mean = args[1]
        running_var = args[2]
        weight = args[3]
        bias = args[4]

        N, C, H, W = input_tensor.shape
        HW = H * W
        n_elements = N * C * HW

        # Ensure params are on the same device as input
        device = input_tensor.device
        running_mean = torch.as_tensor(running_mean, device=device)
        running_var = torch.as_tensor(running_var, device=device)
        weight = torch.as_tensor(weight, device=device)
        bias = torch.as_tensor(bias, device=device)

        output = torch.empty_like(input_tensor)

        BLOCK_SIZE = 2048
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        bn_kernel[grid](
            input_tensor, output,
            running_mean, running_var, weight, bias,
            C, HW, n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output
    else:
        # Avg pool 2d (kernel=2, stride=2, ceil_mode=True)
        x = args[0]
        N, C, H, W = x.shape
        H_out = (H + 1) // 2
        W_out = (W + 1) // 2
        HW_out = H_out * W_out
        n_elements = N * C * HW_out

        output = torch.empty((N, C, H_out, W_out), dtype=x.dtype, device=x.device)

        BLOCK_SIZE = 1024
        grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

        pool_kernel[grid](
            x, output,
            H * W, W, HW_out, W_out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return output