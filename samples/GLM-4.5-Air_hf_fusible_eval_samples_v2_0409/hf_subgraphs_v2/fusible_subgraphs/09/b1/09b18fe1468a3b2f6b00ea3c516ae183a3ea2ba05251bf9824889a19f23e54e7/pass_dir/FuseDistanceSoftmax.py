import torch
import triton
import triton.language as tl

def pattern(in_1, in_2, in_3):
    """Pattern: distance calculation + softmax for attention mechanism"""
    tmp_1 = in_1 - in_2
    tmp_2 = tmp_1.pow(2)
    tmp_3 = tmp_2.sum(dim=3)
    tmp_4 = in_3 * tmp_3
    tmp_5 = torch.nn.functional.softmax(tmp_4, dim=2)
    return tmp_5

def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)

@triton.jit
def fused_distance_softmax_kernel(
    in1_ptr, in2_ptr, scale_ptr,
    out_ptr,
    n1, n2, n3, n4,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for distance + softmax computation"""
    # Calculate program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    codeword_idx = tl.program_id(2)
    
    # Get pointers for current batch/sequence/codeword
    in1_base = in1_ptr + batch_idx * n2 * n3 * n4 + seq_idx * n3 * n4
    in2_base = in2_ptr + codeword_idx * n4
    out_base = out_ptr + batch_idx * n2 * n3 + seq_idx * n3 + codeword_idx
    
    # Load scale for this codeword
    scale = tl.load(scale_ptr + codeword_idx, allow_other=False)
    
    # Load full in1 vector for this position (4096 feature dimensions)
    offsets = tl.arange(0, n4)
    in1 = tl.load(in1_base + offsets, mask=offsets < n4, other=0.0)
    in2 = tl.load(in2_base + offsets, mask=offsets < n4, other=0.0)
    
    # Compute squared distance: (in1 - in2)²
    diff = in1 - in2
    sq_dist = diff * diff
    
    # Sum along feature dimension (already reduced in per-thread computation)
    # Since we're computing for single codeword, this is the reduced distance
    reduced_sq_dist = tl.sum(sq_dist)
    
    # Apply scale and convert to softmax form (exp(x)) for later normalization
    attention_val = tl.exp(scale * reduced_sq_dist)
    
    # Store the result (will be normalized later)
    tl.store(out_base, attention_val, allow_other=False)

@torch.fx.wrap
def fused_distance_softmax(in1, in2, scale):
    """Wrapper for fused distance + softmax computation"""
    # Get shapes
    batch, seq, codewords, feats = in1.shape
    scale_shape = scale.shape
    
    # Output will be [batch, seq, codewords]
    output = torch.empty((batch, seq, codewords), dtype=in1.dtype, device=in1.device)
    
    # Launch kernel for each batch sequence pair
    # Grid: [batch, seq, codewords]
    grid = (batch, seq, codewords)
    
    # Use smaller block size for better memory efficiency
    BLOCK_SIZE = 256
    
    fused_distance_softmax_kernel[grid](
        in1_ptr=in1,
        in2_ptr=in2,
        scale_ptr=scale,
        out_ptr=output,
        n1=batch, n2=seq, n3=codewords, n4=feats,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Apply softmax normalization across codeword dimension
    # Using torch.softmax since it's optimized for this operation
    softmax_output = torch.nn.functional.softmax(output, dim=2)
    
    return softmax_output

def replacement_func():
    return fused_distance_softmax