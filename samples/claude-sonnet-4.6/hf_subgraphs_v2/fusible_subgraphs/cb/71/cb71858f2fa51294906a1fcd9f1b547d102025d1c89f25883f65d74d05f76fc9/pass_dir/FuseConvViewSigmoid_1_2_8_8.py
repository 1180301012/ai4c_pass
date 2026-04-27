import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: conv2d(in_2, in_1, in_0) -> view(1,2,8,8) -> sigmoid
#
# Shapes (from weight_meta.py):
#   in_2  : [1, 2, 1, 8]   – activation  (CUDA)
#   in_1  : [128, 2, 1, 8] – weight      (CPU, moved to CUDA in wrapper)
#   in_0  : [128]           – bias        (CPU, moved to CUDA in wrapper)
#
# The conv2d with stride=(1,1) pad=(0,0) dil=(1,1) groups=1 reduces
# spatial dims to 1×1, so output is [1, 128, 1, 1].
# After view → [1, 2, 8, 8] (128 elements in the same order) → sigmoid.
#
# Equivalent linear: out[o] = sigmoid(bias[o] + dot(in2_flat, in1_flat[o]))
#   where in2_flat  has shape [16]
#         in1_flat  has shape [128, 16]
# ---------------------------------------------------------------------------


def pattern(in_2, in_1, in_0):
    conv = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    y = conv.view(1, 2, 8, 8)
    z = y.sigmoid()
    return z


def replacement_args(in_2, in_1, in_0):
    # Pad to 3 tensor args + route string so dispatch signature matches
    return (in_2, in_1, in_0, "conv_sigmoid")


# ---------------------------------------------------------------------------
# Triton kernel
# Each program (block) handles BLOCK_OUT output elements.
# With BLOCK_OUT=128 and grid=(1,) the single CTA covers all 128 outputs.
# ---------------------------------------------------------------------------

@triton.jit
def _conv_view_sigmoid_kernel(
    in2_ptr,               # [16]  input activation (flattened)
    in1_ptr,               # [128, 16] weight matrix (row-major)
    in0_ptr,               # [128] bias
    out_ptr,               # [128] output (maps to [1,2,8,8])
    N_IN:   tl.constexpr,  # 16
    N_OUT:  tl.constexpr,  # 128
    BLOCK_OUT: tl.constexpr,  # tile size for output channels
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    o_start = pid * BLOCK_OUT
    o_offs = o_start + tl.arange(0, BLOCK_OUT)
    mask = o_offs < N_OUT

    # Accumulate dot-product in fp32
    acc = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    for j in range(N_IN):
        # Scalar load from in2, broadcast across o_offs
        x_val = tl.load(in2_ptr + j).to(tl.float32)
        # Strided load from in1: element [o, j] = in1_ptr[o*N_IN + j]
        w_vals = tl.load(
            in1_ptr + o_offs * N_IN + j, mask=mask, other=0.0
        ).to(tl.float32)
        acc += x_val * w_vals

    bias = tl.load(in0_ptr + o_offs, mask=mask, other=0.0).to(tl.float32)
    acc += bias

    # Sigmoid: 1 / (1 + exp(-x))
    result_f32 = 1.0 / (1.0 + tl.exp(-acc))

    # Cast to the correct output dtype
    if IS_BF16:
        result = result_f32.to(tl.bfloat16)
    else:
        result = result_f32.to(tl.float16)

    tl.store(out_ptr + o_offs, result, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper  (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def conv_view_sigmoid(in_2, in_1, in_0):
    """
    Drop-in replacement for:
        out = torch.conv2d(in_2, in_1, in_0, (1,1),(0,0),(1,1),1)
              .view(1,2,8,8).sigmoid()
    """
    device = in_2.device
    in_1_dev = in_1.to(device)
    in_0_dev = in_0.to(device)

    N_IN   = 16   # 2 * 1 * 8
    N_OUT  = 128
    BLOCK_OUT = 128

    out = torch.empty((1, 2, 8, 8), dtype=in_2.dtype, device=device)
    is_bf16 = (in_2.dtype == torch.bfloat16)

    _conv_view_sigmoid_kernel[(1,)](
        in_2.contiguous(),
        in_1_dev.contiguous(),
        in_0_dev.contiguous(),
        out,
        N_IN=N_IN,
        N_OUT=N_OUT,
        BLOCK_OUT=BLOCK_OUT,
        IS_BF16=is_bf16,
        num_warps=4,
    )

    return out


def replacement_func():
    from pass_dir.shared_dispatch import _dispatch
    return _dispatch