import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.ops.aten.gelu.default(in_0)
    tmp_1 = torch.ops.aten.mean.dim(tmp_0, [2, 3], keepdim=True)
    return (tmp_0, tmp_1)


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=3),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_gelu_mean_kernel(
    input_ptr, gelu_out_ptr, mean_ptr,
    C: tl.int32,
    HW: tl.int32,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (b, c) pair and processes all H*W elements
    pid = tl.program_id(0)
    base = pid * HW

    acc = 0.0

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW

        flat_offsets = base + offsets

        # Load input and upcast to float32 for precision
        x = tl.load(input_ptr + flat_offsets, mask=mask, other=0.0)
        x_f32 = x.to(tl.float32)

        # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
        sqrt2 = 1.4142135623730951
        gelu_f32 = x_f32 * 0.5 * (1.0 + tl.math.erf(x_f32 / sqrt2))

        # Store GELU output (Triton handles dtype conversion to output pointer type)
        tl.store(gelu_out_ptr + flat_offsets, gelu_f32, mask=mask)

        # Accumulate sum for mean computation
        # gelu(0.0) = 0.0, so masked positions don't contribute
        acc += tl.sum(gelu_f32)

    # Store mean = sum / (H * W)
    # Triton handles conversion from float32 to mean_ptr's dtype
    mean_val = acc / HW
    tl.store(mean_ptr + pid, mean_val)


@torch.fx.wrap
def fused_gelu_mean(in_0):
    B, C, H, W = in_0.size()
    HW = H * W

    gelu_out = torch.empty_like(in_0)
    # Create mean output directly in the final shape (B, C, 1, 1) to avoid reshape
    mean_out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    num_programs = B * C

    fused_gelu_mean_kernel[(num_programs,)](
        input_ptr=in_0,
        gelu_out_ptr=gelu_out,
        mean_ptr=mean_out,
        C=C,
        HW=HW,
    )

    return (gelu_out, mean_out)


def replacement_func():
    return fused_gelu_mean