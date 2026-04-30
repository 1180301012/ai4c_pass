import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused scale-subtract + split + squeeze + contiguous
#
# Inputs:
#   in_0  : int64  [B, S, 1]  -- scale factor base
#   in_1  : fp16/bf16 [B, S, 2] -- two-channel value tensor
# Outputs:
#   out0  : same dtype as in_1  [B, S]   -- in_1[:,:,0] - in_0*1e6
#   out1  : same dtype as in_1  [B, S]   -- in_1[:,:,1] - in_0*1e6
#
# One CTA processes all N = B*S elements in a single pass.
# ---------------------------------------------------------------------------

@triton.jit
def fused_scale_split_squeeze_kernel(
    in_0_ptr,            # int64 pointer, stride S (last dim of [B,S,1] = 1)
    in_1_ptr,            # float pointer, stride 2 (channels)
    out0_ptr,            # float pointer, stride S
    out1_ptr,            # float pointer, stride S
    N,                   # total elements = B * S
):
    # BLOCK_SIZE hardcoded as constexpr: covers N<=32 with one CTA
    BLOCK_SIZE = 32
    pid = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = pid < N

    # Load int64 scale values from in_0  (shape [..., 1] → stride S per row)
    scale = tl.load(in_0_ptr + pid, mask=mask, other=0)

    # Load two fp16/bf16 channels from in_1  (stride 2 → even/odd flat indices)
    val0 = tl.load(in_1_ptr + pid * 2,     mask=mask, other=0.0)
    val1 = tl.load(in_1_ptr + pid * 2 + 1, mask=mask, other=0.0)

    # Compute in float32 for precision, cast back to original dtype
    fval0 = val0.to(tl.float32) - scale.to(tl.float32) * 1000000.0
    fval1 = val1.to(tl.float32) - scale.to(tl.float32) * 1000000.0

    # Store to output buffers
    tl.store(out0_ptr + pid, fval0.to(val0.dtype), mask=mask)
    tl.store(out1_ptr + pid, fval1.to(val1.dtype), mask=mask)


@torch.fx.wrap
def fused_scale_split_squeeze(in_0, in_1):
    """
    Fused replacement for:
        tmp_1 = in_0 * 1000000.0
        tmp_2 = in_1 - tmp_1
        split = tmp_2.split(1, dim=-1)
        out0 = split[0].squeeze(-1).contiguous()
        out1 = split[1].squeeze(-1).contiguous()
    """
    # Move in_0 to the same device as in_1 (in_0 lives on CPU in the original model)
    in_0_dev = in_0.to(in_1.device)

    B = in_1.shape[0]
    S = in_1.shape[1]
    N = B * S          # total (b, s) pairs to process

    # Output tensors: [B, S] (matching squeezed+contiguous output)
    out0 = torch.empty((B, S), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((B, S), dtype=in_1.dtype, device=in_1.device)

    # Single CTA: ceil(N / BLOCK_SIZE) = ceil(17/32) = 1
    grid = ((N + 31) // 32,)

    fused_scale_split_squeeze_kernel[grid](
        in_0_dev, in_1, out0, out1,
        N,
    )

    return (out0, out1)


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim=-1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_scale_split_squeeze