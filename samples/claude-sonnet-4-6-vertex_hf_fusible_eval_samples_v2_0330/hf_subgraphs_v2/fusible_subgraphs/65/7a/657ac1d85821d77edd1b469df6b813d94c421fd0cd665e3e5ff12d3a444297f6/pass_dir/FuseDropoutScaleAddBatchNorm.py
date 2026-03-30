import torch
import triton
import triton.language as tl


def pattern(conv_out, scale, residual, bn_mean, bn_var, bn_weight, bn_bias):
    """
    Fuse: dropout(p=0,training=False) + element-wise scale multiply +
          residual add + batch_norm inference.

    Matches (in the model):
        tmp_8  = dropout(conv_out, 0.0, False, False)   # identity
        tmp_9  = tmp_8 * scale                          # layer-scale
        tmp_10 = residual + tmp_9                       # residual add
        tmp_11 = batch_norm(tmp_10, bn_mean, bn_var,
                            bn_weight, bn_bias,
                            False, 0.1, 1e-05)          # inference BN
        return (tmp_11, tmp_10)
    """
    dropped = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    scaled = dropped * scale
    added = residual + scaled
    normed = torch.nn.functional.batch_norm(
        added, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 1e-05
    )
    return (normed, added)


def replacement_args(conv_out, scale, residual, bn_mean, bn_var, bn_weight, bn_bias):
    return (conv_out, scale, residual, bn_mean, bn_var, bn_weight, bn_bias)


# ---------------------------------------------------------------------------
# Triton kernel: fused scale-mul + residual-add + BN-inference
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 512},  num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16),
        triton.Config({"BLOCK_SIZE": 2048}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=8,  num_stages=2),
        triton.Config({"BLOCK_SIZE": 4096}, num_warps=16, num_stages=2),
        triton.Config({"BLOCK_SIZE": 8192}, num_warps=16, num_stages=2),
    ],
    key=["N", "HW"],
)
@triton.jit
def _fused_kernel(
    conv_out_ptr,   # [B, C, H, W]  – conv output (also serves as dropout input)
    scale_ptr,      # [C, 1, 1]     – layer-scale gamma
    residual_ptr,   # [B, C, H, W]  – residual branch
    bn_mean_ptr,    # [C]           – BN running mean
    bn_var_ptr,     # [C]           – BN running var
    bn_weight_ptr,  # [C]           – BN weight (gamma)
    bn_bias_ptr,    # [C]           – BN bias   (beta)
    added_ptr,      # [B, C, H, W]  – output: residual + scaled conv
    normed_ptr,     # [B, C, H, W]  – output: BN(added)
    N,              # total elements = B*C*H*W
    HW,             # H * W
    C,              # number of channels
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Channel index in NCHW layout: c = (offset // HW) % C
    c = (offsets // HW) % C

    # ------------------------------------------------------------------ load
    # Large streaming tensors – hint L2 evict-first to save cache for params
    conv_val  = tl.load(conv_out_ptr  + offsets, mask=mask, other=0.0,
                        eviction_policy="evict_first").to(tl.float32)
    res_val   = tl.load(residual_ptr  + offsets, mask=mask, other=0.0,
                        eviction_policy="evict_first").to(tl.float32)

    # Per-channel parameters – tiny, keep in L2 (evict_last)
    scale_val  = tl.load(scale_ptr     + c, mask=mask, other=1.0,
                         eviction_policy="evict_last").to(tl.float32)
    mean_val   = tl.load(bn_mean_ptr   + c, mask=mask, other=0.0,
                         eviction_policy="evict_last").to(tl.float32)
    var_val    = tl.load(bn_var_ptr    + c, mask=mask, other=1.0,
                         eviction_policy="evict_last").to(tl.float32)
    weight_val = tl.load(bn_weight_ptr + c, mask=mask, other=1.0,
                         eviction_policy="evict_last").to(tl.float32)
    bias_val   = tl.load(bn_bias_ptr   + c, mask=mask, other=0.0,
                         eviction_policy="evict_last").to(tl.float32)

    # --------------------------------------------------------------- compute
    # layer-scale multiply  (dropout p=0 is identity → skip)
    scaled    = conv_val * scale_val

    # residual addition
    added_val = res_val + scaled

    # batch-norm inference
    inv_std   = tl.rsqrt(var_val + eps)
    normed_val = (added_val - mean_val) * inv_std * weight_val + bias_val

    # ----------------------------------------------------------------- store
    # Triton auto-converts float32 → tensor dtype (fp16/bf16/fp32)
    tl.store(added_ptr  + offsets, added_val,  mask=mask, eviction_policy="evict_first")
    tl.store(normed_ptr + offsets, normed_val, mask=mask, eviction_policy="evict_first")


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_scale_add_batchnorm(
    conv_out, scale, residual, bn_mean, bn_var, bn_weight, bn_bias
):
    B, C, H, W = conv_out.shape
    N  = conv_out.numel()
    HW = H * W

    added  = torch.empty_like(conv_out)
    normed = torch.empty_like(conv_out)

    eps = 1e-5
    grid = lambda meta: ((N + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    _fused_kernel[grid](
        conv_out, scale, residual,
        bn_mean, bn_var, bn_weight, bn_bias,
        added, normed,
        N, HW, C,
        eps=eps,
    )

    return (normed, added)


# -----------------------------------------------------------------------
# replacement_kernel: explicitly unpack the tuple so FX sees TWO
# returning nodes (getitem[0] and getitem[1]) that map 1-to-1 with
# the pattern's two returning nodes (normed, added).
# -----------------------------------------------------------------------
def replacement_kernel(conv_out, scale, residual, bn_mean, bn_var, bn_weight, bn_bias):
    result = fused_scale_add_batchnorm(conv_out, scale, residual, bn_mean, bn_var, bn_weight, bn_bias)
    normed = result[0]
    added  = result[1]
    return (normed, added)


def replacement_func():
    return replacement_kernel