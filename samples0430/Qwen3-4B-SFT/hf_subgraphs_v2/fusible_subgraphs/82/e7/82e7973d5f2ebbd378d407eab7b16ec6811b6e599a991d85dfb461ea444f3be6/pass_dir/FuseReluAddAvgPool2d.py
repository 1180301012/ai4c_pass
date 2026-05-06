import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    # Match add + adaptive_avg_pool2d only (in_0=relu_output, in_1=orig_input).
    # Avoids relu kwargs-normalization issue with ForceArgsTracer.
    # Folds relu_out + in_1 → avgpool in a single fused kernel.
    tmp_1 = in_0 + in_1
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64,  'num_warps': 2, 'num_stages': 1}),
        triton.Config({'BLOCK_HW': 64,  'num_warps': 4, 'num_stages': 1}),
        triton.Config({'BLOCK_HW': 128, 'num_warps': 4, 'num_stages': 1}),
        triton.Config({'BLOCK_HW': 128, 'num_warps': 4, 'num_stages': 2}),
        triton.Config({'BLOCK_HW': 256, 'num_warps': 4, 'num_stages': 1}),
        triton.Config({'BLOCK_HW': 256, 'num_warps': 8, 'num_stages': 1}),
        triton.Config({'BLOCK_HW': 512, 'num_warps': 8, 'num_stages': 1}),
        triton.Config({'BLOCK_HW': 512, 'num_warps': 16, 'num_stages': 1}),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def add_avgpool_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (b, c) pair on axis-0.
    pid = tl.program_id(0)
    b = pid // C
    c = pid % C

    base = b * C * HW + c * HW

    # Accumulate (in_0 + in_1) in float32 for numerical stability
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for i in range(0, HW, BLOCK_HW):
        hw_offs = i + tl.arange(0, BLOCK_HW)
        mask = hw_offs < HW

        x0 = tl.load(in0_ptr + base + hw_offs, mask=mask, other=0.0)
        x1 = tl.load(in1_ptr + base + hw_offs, mask=mask, other=0.0)

        acc += x0.to(tl.float32) + x1.to(tl.float32)

    # Global average pooling: sum / HW
    total = tl.sum(acc) / HW

    # Store result (cast back to input dtype)
    tl.store(out_ptr + b * C + c, total.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def add_avgpool(in_0, in_1):
    B, C, H, W = in_0.shape
    HW = H * W

    # Output shape: [B, C, 1, 1]
    output = torch.empty(B, C, 1, 1, dtype=in_0.dtype, device=in_0.device)

    def grid(meta):
        return (B * C, triton.cdiv(HW, meta['BLOCK_HW']))

    add_avgpool_kernel[grid](
        in_0, in_1, output,
        B, C, HW,
    )

    return output


def replacement_func():
    return add_avgpool