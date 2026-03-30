import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Matches: relu -> batch_norm (inference) -> dropout(p=0)
    in_0: running_mean [C]
    in_1: running_var  [C]
    in_2: bias         [C]
    in_3: weight       [C]
    in_4: input        [N, C]
    """
    tmp_4 = torch.nn.functional.relu(in_4, inplace=False)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.dropout(tmp_5, p=0.0, training=False)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 2},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 1},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 4},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 8},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_N': 1},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 4},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 8},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 16}, num_warps=8, num_stages=3),
    ],
    key=['N', 'C'],
)
@triton.jit
def _relu_bn_fused_kernel(
    x_ptr, out_ptr,
    scale_ptr, shift_ptr,
    N, C,
    BLOCK_N: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Each program handles BLOCK_N consecutive rows of the [N, C] tensor.
    scale and shift are precomputed BN parameters (per-channel, float32).
    Computes: out = max(x, 0) * scale + shift
    """
    pid = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_C)
    col_mask = col_offsets < C

    # Load per-channel scale and shift once per program (cached across rows)
    scale = tl.load(scale_ptr + col_offsets, mask=col_mask, other=1.0)
    shift = tl.load(shift_ptr + col_offsets, mask=col_mask, other=0.0)

    # Process BLOCK_N rows
    for i in range(BLOCK_N):
        row = pid * BLOCK_N + i
        mask = col_mask & (row < N)

        x = tl.load(x_ptr + row * C + col_offsets, mask=mask, other=0.0)
        # ReLU
        x = tl.maximum(x, 0.0)
        # BatchNorm inference: x * scale + shift
        x = x * scale + shift
        # Dropout(p=0, training=False) is identity — skipped
        tl.store(out_ptr + row * C + col_offsets, x, mask=mask)


@torch.fx.wrap
def relu_bn_dropout_fused(in_0, in_1, in_2, in_3, in_4):
    """
    Fused ReLU + BatchNorm(inference) + Dropout(p=0) replacement.

    in_0: running_mean  [C]  (CPU)
    in_1: running_var   [C]  (CPU)
    in_2: bias          [C]  (CPU)
    in_3: weight        [C]  (CPU)
    in_4: input         [N, C]  (CUDA)
    """
    device = in_4.device
    orig_dtype = in_4.dtype
    N, C = in_4.shape

    # Precompute scale = weight / sqrt(var + eps)  and
    #           shift  = bias  - mean * scale
    # all in float32 on CPU (tiny, cheap), then transfer to GPU.
    eps = 1e-5
    mean_f32   = in_0.float()
    var_f32    = in_1.float()
    weight_f32 = in_3.float()
    bias_f32   = in_2.float()

    inv_std  = 1.0 / (var_f32 + eps).sqrt()
    scale_cpu = weight_f32 * inv_std
    shift_cpu = bias_f32 - mean_f32 * scale_cpu

    scale = scale_cpu.to(device=device, dtype=torch.float32)
    shift = shift_cpu.to(device=device, dtype=torch.float32)

    # Upcast input to float32 for accurate BN computation
    x_f32  = in_4.to(dtype=torch.float32)
    out_f32 = torch.empty((N, C), device=device, dtype=torch.float32)

    BLOCK_C = triton.next_power_of_2(C)   # 128 for C=128
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_N']),)

    _relu_bn_fused_kernel[grid](
        x_f32, out_f32, scale, shift,
        N, C,
        BLOCK_C=BLOCK_C,
    )

    # Cast back to original dtype (float16 / bfloat16 / float32)
    if orig_dtype != torch.float32:
        return out_f32.to(dtype=orig_dtype)
    return out_f32


def replacement_func():
    return relu_bn_dropout_fused