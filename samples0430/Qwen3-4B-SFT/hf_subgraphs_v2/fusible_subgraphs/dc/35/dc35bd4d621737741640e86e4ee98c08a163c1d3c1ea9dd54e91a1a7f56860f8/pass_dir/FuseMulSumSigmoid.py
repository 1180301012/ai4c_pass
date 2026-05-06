import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
    ],
    key=[],
)
@triton.jit
def fused_mul_sum_sigmoid_kernel(
    in0_ptr, in1_ptr, out_ptr,
    N, B,
    C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE spatial elements for one batch item
    pid = tl.program_id(0)
    n   = pid // (B // BLOCK_SIZE)
    b   = pid %  (B // BLOCK_SIZE)

    spatial_start = b * BLOCK_SIZE
    offsets = spatial_start + tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # C, H, W == 64 (pidnet specific); strided load in HW plane
    for c in range(C):
        ptr_base = n * C * H * W + c * H * W
        for hw in range(H * W // BLOCK_SIZE):
            in_idx = ptr_base + hw * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            v0 = tl.load(in0_ptr + in_idx).to(tl.float32)
            v1 = tl.load(in1_ptr + in_idx).to(tl.float32)
            acc = acc + v0 * v1

    out_idx = n * H * W + tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + out_idx, tl.sigmoid(acc).to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_mul_sum_sigmoid(in_0, in_1):
    N, C, H, W = in_0.shape
    B = H * W  # contiguous spatial dimension
    out = torch.empty((N, 1, H, W), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (N * (B // meta['BLOCK_SIZE']),)

    fused_mul_sum_sigmoid_kernel[grid](
        in_0, in_1, out,
        N, B,
        C, H, W,
    )
    return out


def pattern(in_0, in_1):
    tmp_0 = in_1 * in_0
    tmp_1 = torch.sum(tmp_0, dim=1)
    tmp_2 = tmp_1.unsqueeze(1)
    tmp_3 = torch.sigmoid(tmp_2)
    return tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_mul_sum_sigmoid