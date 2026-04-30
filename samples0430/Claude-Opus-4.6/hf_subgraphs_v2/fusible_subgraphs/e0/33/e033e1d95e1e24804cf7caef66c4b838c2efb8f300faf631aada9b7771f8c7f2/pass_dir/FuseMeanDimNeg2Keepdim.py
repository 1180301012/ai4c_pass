import torch
import triton
import triton.language as tl


def pattern(in_2):
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4


def replacement_args(in_2):
    return (in_2,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 256, 'BLOCK_F': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 512, 'BLOCK_F': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_F': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 2048, 'BLOCK_F': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_R': 4096, 'BLOCK_F': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_R': 256, 'BLOCK_F': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 512, 'BLOCK_F': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_F': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 256, 'BLOCK_F': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_R': 512, 'BLOCK_F': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_F': 64}, num_warps=4, num_stages=2),
    ],
    key=['R', 'F'],
)
@triton.jit
def mean_dim1_kernel(
    input_ptr, output_ptr,
    B, R: tl.constexpr, F: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_f = tl.program_id(1)

    f_offsets = pid_f * BLOCK_F + tl.arange(0, BLOCK_F)
    f_mask = f_offsets < F

    acc = tl.zeros([BLOCK_F], dtype=tl.float32)

    base = pid_b * R * F

    for r_start in range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        offsets = base + r_offsets[:, None] * F + f_offsets[None, :]
        mask = (r_offsets[:, None] < R) & f_mask[None, :]
        vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=0)

    result = acc / R

    out_offsets = pid_b * F + f_offsets
    tl.store(output_ptr + out_offsets, result.to(output_ptr.dtype.element_ty), mask=f_mask)


@torch.fx.wrap
def mean_dim_neg2_keepdim(in_2):
    B = in_2.shape[0]
    R = in_2.shape[1]
    F = in_2.shape[2]

    output = torch.empty((B, 1, F), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (B, (F + meta['BLOCK_F'] - 1) // meta['BLOCK_F'])

    mean_dim1_kernel[grid](
        in_2, output,
        B, R, F,
    )

    return output


def replacement_func():
    return mean_dim_neg2_keepdim