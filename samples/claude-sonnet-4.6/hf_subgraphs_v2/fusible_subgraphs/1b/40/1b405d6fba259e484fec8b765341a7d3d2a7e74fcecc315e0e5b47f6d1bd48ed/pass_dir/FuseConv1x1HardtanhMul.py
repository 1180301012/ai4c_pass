import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors the exact computation in model.py (all six graph variants)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.hardtanh(in_3, 0.0, 6.0, False)
    tmp_4 = tmp_3 * conv2d
    return tmp_4


# ---------------------------------------------------------------------------
# Fused Triton kernel – 3-D grid: (N_batch, ceil(HW/BLOCK_S), ceil(C_out/BLOCK_N))
#
# Design (V4):
#  • 3-D grid avoids div/mod to recover batch index.
#  • inp loaded as [BLOCK_S, BLOCK_K] (k inner) → WMMA-aligned for tensor cores.
#  • wt loaded as [BLOCK_N, BLOCK_K] (k inner, coalesced).
#  • GEMM: tl.dot(wt, tl.trans(inp)) → acc [BLOCK_N, BLOCK_S] in fp32.
#    Swapping operands gives [BLOCK_N, BLOCK_S] layout so in3/output share
#    the same [BLOCK_N, BLOCK_S] coalesced index (s inner, stride 1).
#  • No tl.trans on result → zero register-shuffle overhead in element-wise path.
#  • N_out, C_in constexpr → compiler folds masks at compile time.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        # small-HW / small-batch
        triton.Config({"BLOCK_S": 32,  "BLOCK_N": 32},  num_stages=2, num_warps=2),
        triton.Config({"BLOCK_S": 32,  "BLOCK_N": 64},  num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 64,  "BLOCK_N": 32},  num_stages=2, num_warps=4),
        triton.Config({"BLOCK_S": 64,  "BLOCK_N": 64},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 64,  "BLOCK_N": 128}, num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_N": 32},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_N": 64},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_N": 64},  num_stages=4, num_warps=4),
        triton.Config({"BLOCK_S": 128, "BLOCK_N": 128}, num_stages=3, num_warps=8),
        triton.Config({"BLOCK_S": 128, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        # large-HW / large-batch
        triton.Config({"BLOCK_S": 256, "BLOCK_N": 32},  num_stages=3, num_warps=4),
        triton.Config({"BLOCK_S": 256, "BLOCK_N": 64},  num_stages=3, num_warps=8),
        triton.Config({"BLOCK_S": 256, "BLOCK_N": 64},  num_stages=4, num_warps=8),
        triton.Config({"BLOCK_S": 256, "BLOCK_N": 128}, num_stages=4, num_warps=8),
        triton.Config({"BLOCK_S": 512, "BLOCK_N": 32},  num_stages=4, num_warps=4),
        triton.Config({"BLOCK_S": 512, "BLOCK_N": 64},  num_stages=4, num_warps=8),
        triton.Config({"BLOCK_S": 512, "BLOCK_N": 64},  num_stages=5, num_warps=8),
        triton.Config({"BLOCK_S": 512, "BLOCK_N": 128}, num_stages=4, num_warps=8),
    ],
    key=["N_batch", "HW"],
)
@triton.jit
def fused_1x1conv_hardtanh_mul_kernel(
    inp_ptr,   # [N_batch, C_in, H, W]  – NCHW, CUDA
    wt_ptr,    # [C_out, C_in, 1, 1]   – wt[n,k] = wt_ptr[n*C_in + k]
    bias_ptr,  # [C_out]
    in3_ptr,   # [N_batch, C_out, H, W] – NCHW, CUDA
    out_ptr,   # [N_batch, C_out, H, W] – NCHW, CUDA  (output)
    N_batch,   # batch size
    HW,        # H * W
    N_out: tl.constexpr,  # C_out (96) – constexpr: folds n_mask for BLOCK_N<=96
    C_in: tl.constexpr,   # C_in  (24) – constexpr: folds k_mask at compile time
    BLOCK_S: tl.constexpr,   # spatial tile (power-of-2)
    BLOCK_N: tl.constexpr,   # output-channel tile (power-of-2)
    BLOCK_K: tl.constexpr,   # power-of-2 >= C_in
):
    # 3-D grid: pid_b = batch, pid_s = spatial tile, pid_n = channel tile
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    pid_n = tl.program_id(2)

    s_start = pid_s * BLOCK_S
    n_start = pid_n * BLOCK_N

    s_offs = s_start + tl.arange(0, BLOCK_S)   # [BLOCK_S]  spatial (hw) offsets
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]  output-channel offsets
    k_offs = tl.arange(0, BLOCK_K)             # [BLOCK_K]  input-channel offsets

    s_mask = s_offs < HW
    n_mask = n_offs < N_out
    k_mask = k_offs < C_in

    inp_base = pid_b * C_in * HW
    io_base  = pid_b * N_out * HW

    # -----------------------------------------------------------------------
    # Input: [BLOCK_S, BLOCK_K]
    #   inp[s, k] = inp_ptr[ inp_base + k*HW + s ]
    # This is the WMMA A-matrix aligned layout: for fixed k, threads read
    # consecutive s values (stride 1 in memory) → coalesced with tensor cores.
    # -----------------------------------------------------------------------
    inp_idx  = inp_base + k_offs[None, :] * HW + s_offs[:, None]  # [BLOCK_S, BLOCK_K]
    inp_mask = s_mask[:, None] & k_mask[None, :]
    inp = tl.load(inp_ptr + inp_idx, mask=inp_mask, other=0.0)

    # -----------------------------------------------------------------------
    # Weight: [BLOCK_N, BLOCK_K] – coalesced (k = inner dim, stride 1)
    #   wt[n, k] = wt_ptr[ n*C_in + k ]
    # -----------------------------------------------------------------------
    wt_idx  = n_offs[:, None] * C_in + k_offs[None, :]  # [BLOCK_N, BLOCK_K]
    wt_mask = n_mask[:, None] & k_mask[None, :]
    wt = tl.load(wt_ptr + wt_idx, mask=wt_mask, other=0.0)

    # -----------------------------------------------------------------------
    # GEMM: wt @ inp.T = [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_S] → [BLOCK_N, BLOCK_S]
    #   acc[n, s] = Σ_k wt[n,k] * inp[s,k]
    # tl.trans(inp) is resolved INSIDE tl.dot with zero register-shuffle cost.
    # Result in [BLOCK_N, BLOCK_S] layout allows coalesced in3/output I/O.
    # -----------------------------------------------------------------------
    acc = tl.dot(wt, tl.trans(inp), out_dtype=tl.float32)  # [BLOCK_N, BLOCK_S]

    # Bias [BLOCK_N] broadcast along S dimension
    bias = tl.load(bias_ptr + n_offs, mask=n_mask, other=0.0)
    acc  = acc + bias[:, None].to(tl.float32)

    # -----------------------------------------------------------------------
    # in3: [BLOCK_N, BLOCK_S] – coalesced (s = inner dim, stride 1)
    #   in3[n, s] = in3_ptr[ io_base + n*HW + s ]
    # No tl.trans needed; result is already in [BLOCK_N, BLOCK_S] layout.
    # -----------------------------------------------------------------------
    io_idx  = io_base + n_offs[:, None] * HW + s_offs[None, :]  # [BLOCK_N, BLOCK_S]
    io_mask = n_mask[:, None] & s_mask[None, :]
    in3_T = tl.load(in3_ptr + io_idx, mask=io_mask, other=0.0)

    # Fused hardtanh + multiply (no transpose required – both [BLOCK_N, BLOCK_S])
    in3_relu6 = tl.minimum(tl.maximum(in3_T.to(tl.float32), 0.0), 6.0)
    result    = in3_relu6 * acc  # [BLOCK_N, BLOCK_S]

    # Store [BLOCK_N, BLOCK_S] – coalesced (s = inner dim, stride 1)
    tl.store(out_ptr + io_idx, result.to(in3_T.dtype), mask=io_mask)


# ---------------------------------------------------------------------------
# Python wrapper – called by the replacement framework
# Only allowed tensor ops: .to() for device/dtype transfer + torch.empty_like
# No .view() / .t() / .contiguous() – those go through blocked aten dispatch
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_1x1conv_hardtanh_mul(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [C_out]                 – may be on CPU
    in_1 : weight [C_out, C_in, 1, 1]    – may be on CPU
    in_2 : input  [N, C_in, H, W]        – CUDA
    in_3 :        [N, C_out, H, W]       – CUDA
    """
    device  = in_2.device
    dtype   = in_2.dtype

    N_batch, C_in, H, W = in_2.shape
    C_out = in_1.shape[0]
    HW    = H * W

    # Move weights to CUDA – keep original layout; wt[n,k]=wt_ptr[n*C_in+k]
    wt   = in_1.to(device=device, dtype=dtype)
    bias = in_0.to(device=device, dtype=dtype)

    out  = torch.empty_like(in_3)

    # BLOCK_K: power-of-2 >= C_in
    BLOCK_K = 32 if C_in <= 32 else 64

    # 3-D grid: (batch, spatial_tiles, channel_tiles)
    grid = lambda meta: (
        N_batch,
        triton.cdiv(HW,   meta["BLOCK_S"]),
        triton.cdiv(C_out, meta["BLOCK_N"]),
    )

    fused_1x1conv_hardtanh_mul_kernel[grid](
        in_2, wt, bias, in_3, out,
        N_batch, HW,
        C_out,          # N_out constexpr – folds n_mask for BLOCK_N <= 96
        C_in,           # C_in  constexpr – folds k_mask for K=24
        BLOCK_K=BLOCK_K,
    )

    return out


# ---------------------------------------------------------------------------
# Required pass interface
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_1x1conv_hardtanh_mul