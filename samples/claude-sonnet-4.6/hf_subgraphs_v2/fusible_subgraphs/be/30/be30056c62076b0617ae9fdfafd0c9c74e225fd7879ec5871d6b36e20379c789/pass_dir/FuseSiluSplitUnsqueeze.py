import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: silu → split([512, 512, 128], dim=2) → getitem ×3
#          → unsqueeze(2) on chunk-2
# (in_0[None, None, :] is a free view; left out of the pattern)
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel: SiLU + scatter-write into 3 output buffers
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def silu_split_triton_kernel(
    in_ptr,
    out0_ptr,   # [B, T, 512]
    out1_ptr,   # [B, T, 512]
    out2_ptr,   # [B, T, 128]
    n_elements,
    T,          # sequence length, typically 17
    C,          # input channel width = 1152
    BLOCK_SIZE: tl.constexpr,
):
    C0: tl.constexpr = 512
    C1: tl.constexpr = 512
    C2: tl.constexpr = 128
    C01: tl.constexpr = 1024   # C0 + C1

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode (b, t, c) from flat offset
    c_full = offsets % C
    bt     = offsets // C
    t      = bt % T
    b      = bt // T

    # Load + SiLU in fp32
    x      = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    x_f32  = x.to(tl.float32)
    x_silu = x_f32 * tl.sigmoid(x_f32)
    x_out  = x_silu.to(x.dtype)

    # ── chunk 0: c in [0, 512) → out0 ──────────────────────────────────────
    m0   = mask & (c_full < C0)
    idx0 = b * T * C0 + t * C0 + c_full
    tl.store(out0_ptr + idx0, x_out, mask=m0)

    # ── chunk 1: c in [512, 1024) → out1 ───────────────────────────────────
    m1   = mask & (c_full >= C0) & (c_full < C01)
    idx1 = b * T * C1 + t * C1 + (c_full - C0)
    tl.store(out1_ptr + idx1, x_out, mask=m1)

    # ── chunk 2: c in [1024, 1152) → out2 ──────────────────────────────────
    m2   = mask & (c_full >= C01)
    idx2 = b * T * C2 + t * C2 + (c_full - C01)
    tl.store(out2_ptr + idx2, x_out, mask=m2)


@torch.fx.wrap
def silu_split_wrapper(in_1):
    B, T, C = in_1.shape           # e.g. [512, 17, 1152]

    out0 = torch.empty((B, T, 512), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((B, T, 512), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((B, T, 128), dtype=in_1.dtype, device=in_1.device)

    n_elements = in_1.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    silu_split_triton_kernel[grid](
        in_1, out0, out1, out2,
        n_elements, T, C,
    )

    # out2 unsqueeze is a zero-copy view
    out2_unsqueezed = out2.unsqueeze(2)   # [B, T, 1, 128]

    return (out0, out2_unsqueezed, out1)


def replacement_func():
    return silu_split_wrapper