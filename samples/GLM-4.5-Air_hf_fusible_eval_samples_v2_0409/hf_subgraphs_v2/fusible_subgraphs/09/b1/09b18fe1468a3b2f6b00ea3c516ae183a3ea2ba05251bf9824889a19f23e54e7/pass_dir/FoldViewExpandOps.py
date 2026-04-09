import torch
import triton
import triton.language as tl

def pattern(in_0, in_4):
    """Pattern: view + unsqueeze + expand operations for tensor broadcasting"""
    tmp_6 = in_0.view((1, 1, 32, 512))
    tmp_7 = in_4.unsqueeze(2)
    tmp_8 = tmp_7.expand((1, 4096, 32, 512))
    return tmp_8

def replacement_args(in_0, in_4):
    return (in_0, in_4)

@triton.jit
def optimized_broadcast_kernel(
    in0_ptr, in4_ptr,
    out_ptr,
    n1, n2, n3, n4,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for tensor broadcasting operations"""
    # Calculate program indices
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    codeword_idx = tl.program_id(2)
    feat_idx = tl.program_id(3)
    
    # For in_0: [32, 512] -> [1, 1, 32, 512]
    # Load the corresponding element from in_0
    in0_offset = codeword_idx * n4 + feat_idx
    in0_val = tl.load(in0_ptr + in0_offset, mask=in0_offset < (n3 * n4), other=0.0)
    
    # For in_4: [1, 4096, 512] -> expand to [1, 4096, 32, 512]
    # Load the corresponding element from in_4
    in4_offset = seq_idx * n4 + feat_idx
    in4_val = tl.load(in4_ptr + in4_offset, mask=in4_offset < (n2 * n4), other=0.0)
    
    # The output is the expansion of in_4, so we just assign the loaded value
    # Since we're broadcasting in_0 dimensions [1,1,32,512] and expanding in_4 [1,4096,1,512]
    # to [1,4096,32,512], the output element is simply in4_val
    out_offset = batch_idx * n2 * n3 * n4 + seq_idx * n3 * n4 + codeword_idx * n4 + feat_idx
    tl.store(out_ptr + out_offset, in4_val, allow_other=False)

@torch.fx.wrap
def optimized_broadcast(in0, in4):
    """Wrapper for optimized tensor broadcasting"""
    # Get input shapes
    in0_shape = in0.shape  # [32, 512]
    in4_shape = in4.shape  # [1, 4096, 512]
    
    # Output shape: [1, 4096, 32, 512]
    batch, seq, codewords, feats = 1, in4_shape[1], in0_shape[0], in0_shape[1]
    
    output = torch.empty((batch, seq, codewords, feats), dtype=in4.dtype, device=in4.device)
    
    # Launch kernel for each element in the expanded tensor
    # Grid: [batch, seq, codewords, feats]
    grid = (batch, seq, codewords, feats)
    
    BLOCK_SIZE = 256
    
    optimized_broadcast_kernel[grid](
        in0_ptr=in0,
        in4_ptr=in4,
        out_ptr=output,
        n1=batch, n2=seq, n3=codewords, n4=feats,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_broadcast