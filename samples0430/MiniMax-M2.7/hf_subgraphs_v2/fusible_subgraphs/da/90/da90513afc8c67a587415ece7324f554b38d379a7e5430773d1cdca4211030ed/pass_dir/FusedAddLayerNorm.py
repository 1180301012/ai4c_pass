import torch
import triton
import triton.language as tl


@triton.jit
def fused_add_ln_kernel(
    in_2_ptr, in_3_ptr, in_1_ptr, in_0_ptr,
    out_ptr,
    stride_batch, stride_seq, stride_feat,
    n_batch, n_seq, n_feat,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get the sequence dimension block this program is responsible for
    # Each program processes one sequence position for all features
    seq_pid = tl.program_id(0)
    batch_pid = tl.program_id(1)
    
    # Calculate offsets
    feat_offsets = tl.arange(0, BLOCK_SIZE)
    mask = feat_offsets < n_feat
    
    # Base offset for this batch and sequence
    base_offset = batch_pid * stride_batch + seq_pid * stride_seq
    
    # Load feature data for this batch and sequence position
    # Shape: [n_batch, n_seq, n_feat], normalize over last dim (n_feat)
    x1 = tl.load(in_2_ptr + base_offset + feat_offsets * stride_feat, mask=mask, other=0.0).to(tl.float32)
    x2 = tl.load(in_3_ptr + base_offset + feat_offsets * stride_feat, mask=mask, other=0.0).to(tl.float32)
    
    # Sum the two inputs
    x_sum = x1 + x2
    
    # Compute statistics over the feature dimension
    mean = x_sum / n_feat
    
    # Variance: E[(x - mean)^2]
    diff = x_sum - mean
    sq_diff = diff * diff
    variance = sq_diff / n_feat
    inv_std = 1.0 / tl.sqrt(variance + eps)
    
    # Normalize
    norm = (x_sum - mean) * inv_std
    
    # Load weight and bias
    weight = tl.load(in_1_ptr + feat_offsets, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(in_0_ptr + feat_offsets, mask=mask, other=0.0).to(tl.float32)
    
    # Scale and shift
    out = norm * weight + bias
    
    # Store result
    out_offset = batch_pid * n_seq * n_feat + seq_pid * n_feat + feat_offsets
    tl.store(out_ptr + out_offset, out, mask=mask)


@torch.fx.wrap
def fused_add_ln_dispatch(in_0, in_1, in_2, in_3, route=""):
    """
    Dispatcher for fused addition + layer normalization kernel.
    Routes to the appropriate kernel based on the route string.
    in_0: bias tensor [n_feat]
    in_1: weight tensor [n_feat]
    in_2: input tensor [n_batch, n_seq, n_feat]
    in_3: input tensor [n_batch, n_seq, n_feat]
    route: string to distinguish between different normalized_shape values (not used, same kernel for all)
    Returns: normalized output [n_batch, n_seq, n_feat]
    """
    n_batch, n_seq, n_feat = in_2.shape
    
    # Allocate output
    out = torch.empty_like(in_2)
    
    # Block size should be a power of 2, at least n_feat
    BLOCK_SIZE = min(1024, triton.next_power_of_2(n_feat))
    
    # Grid: (n_seq, n_batch) - each program processes one sequence position
    grid = (n_seq, n_batch)
    
    fused_add_ln_kernel[grid](
        in_2, in_3, in_1, in_0,
        out,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        n_batch, n_seq, n_feat,
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


# Pass for normalized_shape = (768,)
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: in_2 + in_3, then layer_norm with normalized_shape=(768,)
    """
    tmp_2 = in_2 + in_3
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (768,), in_1, in_0, 1e-05)
    return (tmp_3,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "768")


def replacement_func():
    return fused_add_ln_dispatch