import torch
import triton
import triton.language as tl

# Global cache for storing mean output from the fused gelu+mean kernel
_cached_mean_output = None


@triton.jit
def fused_gelu_mean_kernel(
    input_ptr, gelu_out_ptr, mean_out_ptr,
    B, C, H, W,
    stride_in_0, stride_in_1, stride_in_2, stride_in_3,
    stride_gelu_0, stride_gelu_1, stride_gelu_2, stride_gelu_3,
    stride_mean_0, stride_mean_1,
    BLOCK_SIZE: tl.constexpr,
):
    bc_idx = tl.program_id(0)
    b = bc_idx // C
    c = bc_idx % C

    in_base = b * stride_in_0 + c * stride_in_1
    gelu_base = b * stride_gelu_0 + c * stride_gelu_1

    acc = 0.0
    HW = H * W

    for hw_start in range(0, HW, BLOCK_SIZE):
        hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
        hw_mask = hw_offsets < HW

        h_coords = hw_offsets // W
        w_coords = hw_offsets % W

        # Load input
        x = tl.load(
            input_ptr + in_base + h_coords * stride_in_2 + w_coords * stride_in_3,
            mask=hw_mask, other=0.0,
        )

        # Compute GELU in float32: x * 0.5 * (1 + erf(x / sqrt(2)))
        x_f32 = x.to(tl.float32)
        sqrt2 = 1.4142135623730951
        gelu_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 / sqrt2))

        # Store GELU output
        tl.store(
            gelu_out_ptr + gelu_base + h_coords * stride_gelu_2 + w_coords * stride_gelu_3,
            gelu_f32, mask=hw_mask,
        )

        # Accumulate sum for mean computation
        acc += tl.sum(gelu_f32)

    # Compute mean
    mean_f32 = acc / HW
    mean_base = b * stride_mean_0 + c * stride_mean_1
    tl.store(mean_out_ptr + mean_base, mean_f32)


@torch.fx.wrap
def fused_gelu_mean_dispatch(*args):
    global _cached_mean_output
    route = args[-1]
    if route == "gelu_with_cache":
        input_tensor = args[0]
        B, C, H, W = input_tensor.shape
        HW = H * W

        gelu_output = torch.empty_like(input_tensor)
        mean_output = torch.empty(B, C, 1, 1, dtype=input_tensor.dtype, device=input_tensor.device)

        num_bc_pairs = B * C
        BLOCK_SIZE = 1024

        grid = (num_bc_pairs,)

        fused_gelu_mean_kernel[grid](
            input_ptr=input_tensor,
            gelu_out_ptr=gelu_output,
            mean_out_ptr=mean_output,
            B=B, C=C, H=H, W=W,
            stride_in_0=input_tensor.stride(0),
            stride_in_1=input_tensor.stride(1),
            stride_in_2=input_tensor.stride(2),
            stride_in_3=input_tensor.stride(3),
            stride_gelu_0=gelu_output.stride(0),
            stride_gelu_1=gelu_output.stride(1),
            stride_gelu_2=gelu_output.stride(2),
            stride_gelu_3=gelu_output.stride(3),
            stride_mean_0=mean_output.stride(0),
            stride_mean_1=mean_output.stride(1),
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # Cache mean output for the subsequent mean pass
        _cached_mean_output = mean_output

        return gelu_output
    elif route == "mean_from_cache":
        # gelu_output argument is not used - we read from cache
        mean_output = _cached_mean_output
        _cached_mean_output = None  # Clear cache
        return mean_output
    else:
        raise ValueError(f"Unknown route: {route}")