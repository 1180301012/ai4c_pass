import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_ln_kernel_1024(
    in_2_ptr, in_3_ptr, in_1_ptr, in_0_ptr,
    out_ptr,
    stride_batch, stride_seq, stride_feat,
    n_batch, n_seq, n_feat,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    seq_pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    feat_offsets = tl.arange(0, BLOCK_SIZE)
    mask = feat_offsets < n_feat
    
    base_offset = batch_pid * stride_batch + seq_pid * stride_seq
    
    x1 = tl.load(in_2_ptr + base_offset + feat_offsets * stride_feat, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in_3_ptr + base_offset + feat_offsets * stride_feat, mask=mask, other=0.0).to(tl.float32)
    
    x_sum = x1 + x2
    mean = x_sum / n_feat
    
    diff = x_sum - mean
    sq_diff = diff * diff
    variance = sq_diff / n_feat
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    norm = (x_sum - mean) * inv_std
    
    weight = tl.load(in_1_ptr + feat_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(in_0_ptr + feat_offsets, mask=mask, other=0.0).to(tl.float32)
    
    out = norm * weight + bias
    
    out_offset = batch_pid * n_seq * n_feat + seq_pid * n_feat + feat_offsets
    tl.store(out_ptr + out_offset, out, mask=mask)


@torch.fx.wrap
def fused_add_ln_wrapper_1024(in_0, in_1, in_2, in_3, route=""):
    n_batch, n_seq, n_feat = in_2.shape
    out = torch.empty_like(in_2)
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_feat))
    grid = (n_seq, n_batch)
    
    fused_add_ln_kernel_1024[grid](
        in_2, in_3, in_1, in_0,
        out,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        n_batch, n_seq, n_feat,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: in_2 + in_3, then layer_norm with normalized_shape=(1024,)
    """
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (1024,), in_1, in_0, 1e-05)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "1024")


def replacement_func():
    return fused_add_ln_wrapper_1024