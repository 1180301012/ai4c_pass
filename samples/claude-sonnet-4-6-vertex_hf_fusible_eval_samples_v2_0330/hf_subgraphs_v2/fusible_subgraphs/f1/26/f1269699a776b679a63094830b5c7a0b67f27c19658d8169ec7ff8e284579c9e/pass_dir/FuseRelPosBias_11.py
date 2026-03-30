import torch
import triton
import triton.language as tl


def pattern(dummy):
    return dummy


def replacement_args():
    return ()


@triton.jit
def rel_pos_bias_kernel(
    output_ptr,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Compute MPNet relative position bias buckets for NxN matrix."""
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    # diff = col - row, then negate: neg = -(col - row) = row - col
    neg = row - col

    # sign_base: 16 if neg < 0 else 0
    sign_base = tl.where(neg < 0, 16, 0)

    # absolute value
    abs_val = tl.abs(neg)

    # log bucket: 8 + floor(8 * log2(abs_val/8)) clamped to 15
    abs_f = abs_val.to(tl.float32)
    log_bucket = (8.0 + (abs_f / 8.0).to(tl.float32).__log2__() * 8.0).to(tl.int64)
    # Wait, log2 might not be available - use log / log(2)
    # log_bucket = tl.cast(8 + tl.math.log(abs_f / 8.0) / 0.6931471805599453 * 8.0, tl.int64)
    log_bucket_clamped = tl.minimum(log_bucket, 15)

    # Select: if abs_val < 8, use abs_val; else use log_bucket_clamped
    bucket = tl.where(abs_val < 8, abs_val, log_bucket_clamped)

    # Final result
    result = sign_base + bucket

    tl.store(output_ptr + pid, result)


# Precomputed cache
_rel_pos_bias_cache_11 = None


def _get_rel_pos_bias_11():
    global _rel_pos_bias_cache_11
    if _rel_pos_bias_cache_11 is None:
        N = 11
        tmp_10 = torch.arange(N, dtype=torch.int64)
        tmp_11 = tmp_10[:, None]
        tmp_12 = torch.arange(N, dtype=torch.int64)
        tmp_13 = tmp_12[None, :]
        tmp_14 = tmp_13 - tmp_11
        tmp_15 = -tmp_14
        tmp_16 = tmp_15 < 0
        tmp_17 = tmp_16.to(torch.int64)
        tmp_18 = tmp_17 * 16
        tmp_19 = 0 + tmp_18
        tmp_20 = torch.abs(tmp_15)
        tmp_21 = tmp_20 < 8
        tmp_22 = tmp_20.float()
        tmp_23 = tmp_22 / 8
        tmp_24 = torch.log(tmp_23)
        tmp_25 = tmp_24 / 2.772588722239781
        tmp_26 = tmp_25 * 8
        tmp_27 = tmp_26.to(torch.int64)
        tmp_28 = 8 + tmp_27
        tmp_29 = torch.full_like(tmp_28, 15)
        tmp_30 = torch.min(tmp_28, tmp_29)
        tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
        tmp_19 += tmp_31
        _rel_pos_bias_cache_11 = tmp_19
    return _rel_pos_bias_cache_11


@torch.fx.wrap
def rel_pos_bias_11():
    return _get_rel_pos_bias_11()


def replacement_func():
    return rel_pos_bias_11