import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_bn_add_relu_mean_kernel(
    x_ptr,         # in_4: input [N, C, H, W]
    res_ptr,       # in_5: residual [N, C, H, W]
    rm_ptr,        # in_0: running_mean [C]
    rv_ptr,        # in_1: running_var [C]
    weight_ptr,    # in_3: BN weight (gamma) [C]
    bias_ptr,      # in_2: BN bias (beta) [C]
    out_ptr,       # output: [N, C, H, W]
    mean_out_ptr,  # output mean: [N, C] (flattened from [N,C,1,1])
    N, C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) slice
    pid = tl.program_id(0)   # pid = n * C + c
    c_id = pid % C

    # Load per-channel BN parameters, upcast to float32 for numerical stability
    rm = tl.load(rm_ptr + c_id).to(tl.float32)
    rv = tl.load(rv_ptr + c_id).to(tl.float32)
    w  = tl.load(weight_ptr + c_id).to(tl.float32)
    b  = tl.load(bias_ptr + c_id).to(tl.float32)

    # Precompute BN affine constants: y = x * scale + shift
    inv_std = tl.rsqrt(rv + eps)
    scale   = w * inv_std
    shift   = b - rm * scale

    # Base offset for this (n, c) slice in contiguous NCHW layout
    base = pid * HW

    # Accumulator for spatial mean reduction
    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Process all H*W elements in BLOCK_HW-sized tiles
    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW

        # Load inputs (keep in original dtype for memory efficiency)
        x = tl.load(x_ptr   + base + offs, mask=mask, other=0.0)
        r = tl.load(res_ptr  + base + offs, mask=mask, other=0.0)

        # Compute in float32: BN -> residual add -> ReLU
        x_f32  = x.to(tl.float32)
        r_f32  = r.to(tl.float32)
        bn_val = x_f32 * scale + shift
        val    = tl.maximum(r_f32 + bn_val, 0.0)

        # Store ReLU output cast back to original dtype
        tl.store(out_ptr + base + offs, val.to(x.dtype), mask=mask)

        # Accumulate masked values for mean
        acc += tl.where(mask, val, 0.0)

    # Compute spatial mean and store (element at [n, c, 0, 0] has linear index pid)
    mean_val = tl.sum(acc) / HW
    tl.store(mean_out_ptr + pid, mean_val.to(out_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_bn_add_relu_mean(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: running_mean  [C]
    # in_1: running_var   [C]
    # in_2: bias          [C]
    # in_3: weight        [C]
    # in_4: input         [N, C, H, W]
    # in_5: residual      [N, C, H, W]
    N, C, H, W = in_4.shape
    HW = H * W

    out      = torch.empty_like(in_4)
    mean_out = torch.empty((N, C, 1, 1), dtype=in_4.dtype, device=in_4.device)

    # Launch one program per (n, c) slice
    grid = (N * C,)

    fused_bn_add_relu_mean_kernel[grid](
        in_4, in_5,
        in_0, in_1, in_3, in_2,
        out, mean_out,
        N, C, HW,
        eps=1e-05,
    )

    return (out, mean_out)


def replacement_func():
    return fused_bn_add_relu_mean