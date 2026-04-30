import torch
import triton
import triton.language as tl


@triton.jit
def _attn_mask_kernel_n2(
    in0_ptr, in2_ptr, out_ptr,
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    """
    Fused attention mask kernel.
    Computes: out[b, 0, i, j] = (in0[b,j] != 0) AND (j <= in2[i])
    Grid: (B * Nq,)  -- each program handles one (batch, query_pos) pair
    """
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    k_offs = tl.arange(0, BLOCK_N)
    mask = k_offs < Nk
    x = tl.load(in0_ptr + b * Nk + k_offs, mask=mask, other=0)
    pos = tl.load(in2_ptr + k_offs, mask=mask, other=0)
    valid = (x != 0)
    causal = (k_offs <= i)
    out_mask = valid & causal
    tl.store(out_ptr + pid * Nk + k_offs, out_mask, mask=mask)


@triton.jit
def _attn_mask_kernel_n3(
    in0_ptr, in2_ptr, out_ptr,
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    k_offs = tl.arange(0, BLOCK_N)
    mask = k_offs < Nk
    x = tl.load(in0_ptr + b * Nk + k_offs, mask=mask, other=0)
    pos = tl.load(in2_ptr + k_offs, mask=mask, other=0)
    valid = (x != 0)
    causal = (k_offs <= i)
    out_mask = valid & causal
    tl.store(out_ptr + pid * Nk + k_offs, out_mask, mask=mask)


@triton.jit
def _attn_mask_kernel_n64(
    in0_ptr, in2_ptr, out_ptr,
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    k_offs = tl.arange(0, BLOCK_N)
    mask = k_offs < Nk
    x = tl.load(in0_ptr + b * Nk + k_offs, mask=mask, other=0)
    pos = tl.load(in2_ptr + k_offs, mask=mask, other=0)
    valid = (x != 0)
    causal = (k_offs <= i)
    out_mask = valid & causal
    tl.store(out_ptr + pid * Nk + k_offs, out_mask, mask=mask)


@triton.jit
def _attn_mask_kernel_n128(
    in0_ptr, in2_ptr, out_ptr,
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    k_offs = tl.arange(0, BLOCK_N)
    mask = k_offs < Nk
    x = tl.load(in0_ptr + b * Nk + k_offs, mask=mask, other=0)
    pos = tl.load(in2_ptr + k_offs, mask=mask, other=0)
    valid = (x != 0)
    causal = (k_offs <= i)
    out_mask = valid & causal
    tl.store(out_ptr + pid * Nk + k_offs, out_mask, mask=mask)


@triton.jit
def _attn_mask_kernel_n256(
    in0_ptr, in2_ptr, out_ptr,
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    k_offs = tl.arange(0, BLOCK_N)
    mask = k_offs < Nk
    x = tl.load(in0_ptr + b * Nk + k_offs, mask=mask, other=0)
    pos = tl.load(in2_ptr + k_offs, mask=mask, other=0)
    valid = (x != 0)
    causal = (k_offs <= i)
    out_mask = valid & causal
    tl.store(out_ptr + pid * Nk + k_offs, out_mask, mask=mask)


@triton.jit
def _attn_mask_kernel_n512(
    in0_ptr, in2_ptr, out_ptr,
    B, Nq, Nk,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // Nq
    i = pid % Nq
    k_offs = tl.arange(0, BLOCK_N)
    mask = k_offs < Nk
    x = tl.load(in0_ptr + b * Nk + k_offs, mask=mask, other=0)
    pos = tl.load(in2_ptr + k_offs, mask=mask, other=0)
    valid = (x != 0)
    causal = (k_offs <= i)
    out_mask = valid & causal
    tl.store(out_ptr + pid * Nk + k_offs, out_mask, mask=mask)


def _run_attn_mask(in_0, in_2, kernel_fn):
    """
    Generic wrapper for all attention mask kernels.
    in_0: [B, Nk] int64  (attention_mask)
    in_2: [Nq] int64     (cache_position / arange for causal mask)
    Returns: [B, 1, Nq, Nk] bool
    """
    B = in_0.shape[0]
    Nk = in_0.shape[1]
    Nq = in_2.shape[0]
    out = torch.empty((B, 1, Nq, Nk), dtype=torch.bool, device=in_0.device)
    grid = (B * Nq,)
    kernel_fn[grid](
        in_0, in_2, out,
        B, Nq, Nk,
    )
    return out


def _run_attn_mask_n2(in_0, in_2):
    return _run_attn_mask(in_0, in_2, _attn_mask_kernel_n2)


def _run_attn_mask_n3(in_0, in_2):
    return _run_attn_mask(in_0, in_2, _attn_mask_kernel_n3)


def _run_attn_mask_n64(in_0, in_2):
    return _run_attn_mask(in_0, in_2, _attn_mask_kernel_n64)


def _run_attn_mask_n128(in_0, in_2):
    return _run_attn_mask(in_0, in_2, _attn_mask_kernel_n128)


def _run_attn_mask_n256(in_0, in_2):
    return _run_attn_mask(in_0, in_2, _attn_mask_kernel_n256)


def _run_attn_mask_n512(in_0, in_2):
    return _run_attn_mask(in_0, in_2, _attn_mask_kernel_n512)