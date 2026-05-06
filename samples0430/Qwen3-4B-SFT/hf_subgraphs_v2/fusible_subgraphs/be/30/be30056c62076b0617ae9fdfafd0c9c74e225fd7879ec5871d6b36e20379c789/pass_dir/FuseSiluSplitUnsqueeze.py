import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: silu -> split([512, 512, 128], dim=2) -> getitem[2] -> unsqueeze(2)
# All graphs share this exact computation regardless of batch size or dtype.
# ---------------------------------------------------------------------------
def pattern(in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    tmp_6 = tmp_5.unsqueeze(2)
    return (tmp_3, tmp_6, tmp_4)


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: fused SiLU (x * sigmoid(x))
# Works for float32, bfloat16, and float16 in a single constexpr-free kernel.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=2),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def silu_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load; Triton promotes fp16/bf16 to fp32 for sigmoid accuracy
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)

    # SiLU = x * sigmoid(x) = x / (1 + exp(-x))
    fc = tl.math.exp2(-x_f32 * 2.0)
    s = 1.0 / (1.0 + (-fc).to(tl.float32))
    out_f32 = x_f32 * s

    # Cast back to original dtype before storing
    tl.store(out_ptr + offsets, out_f32.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper: compute silu(in_1), then return views as split + unsqueeze
# (split and unsqueeze are zero-copy view operations in PyTorch)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_silu_split_unsqueeze(in_1):
    B = in_1.shape[0]
    T = in_1.shape[1]          # = 17
    S = in_1.shape[2]          # = 1152
    N = B * T * S              # total elements

    # Allocate output with same shape as silu input
    out = torch.empty_like(in_1)

    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)
    silu_kernel[grid](in_1, out, N)

    # split_out0 [B, T, 512]
    split_out0 = out[:, :, :512]
    # split_out1 [B, T, 512]
    split_out1 = out[:, :, 512:1024]
    # split_out2 [B, T, 128] -> view as [B, T, 1, 128] via unsqueeze(2)
    split_out2 = out[:, :, 1024:]
    # tmp_6 = split_out2.unsqueeze(2)  [B, T, 1, 128]
    tmp_6 = split_out2.unsqueeze(2)

    return split_out0, tmp_6, split_out1


# ---------------------------------------------------------------------------
# replacement_func: return the wrapper (zero-argument, returns callable)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_silu_split_unsqueeze