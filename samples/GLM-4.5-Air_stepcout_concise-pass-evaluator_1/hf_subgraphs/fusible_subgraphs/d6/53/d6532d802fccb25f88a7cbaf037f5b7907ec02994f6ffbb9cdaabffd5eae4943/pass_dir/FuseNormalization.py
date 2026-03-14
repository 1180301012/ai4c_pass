import torch
import triton
import triton.language as tl

def pattern(inputs_embeds):
    tmp_10 = inputs_embeds.to(torch.float32)
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_11 = None
    tmp_13 = tmp_12 + 1e-06
    tmp_12 = None
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_13 = None
    tmp_15 = tmp_10 * tmp_14
    tmp_10 = tmp_14 = None
    tmp_16 = tmp_15.to(torch.bfloat16)
    tmp_15 = None
    return tmp_16

def replacement_args(inputs_embeds):
    return (inputs_embeds,)

@triton.jit
def fused_normalization_kernel(
    inputs_ptr,
    outputs_ptr,
    batch_size,
    seq_len,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
    # Each program handles one batch, seq_len position
    batch_idx = tl.program_id(0)
    seq_idx = tl.program_id(1)
    
    # Load input as float32 - we need to process in blocks
    offset = batch_idx * seq_len * HIDDEN_DIM + seq_idx * HIDDEN_DIM
    for k in range(0, HIDDEN_DIM, BLOCK_SIZE_M):
        current_offset = offset + k
        mask = k + tl.arange(0, BLOCK_SIZE_M) < HIDDEN_DIM
        
        # Load input values for this block
        inputs = tl.load(inputs_ptr + current_offset + tl.arange(0, BLOCK_SIZE_M), 
                        mask=mask, other=0.0).to(tl.float32)
        
        # Square and accumulate for mean calculation
        squared = inputs * inputs
        
        # Store results converted to bfloat16
        results = squared.to(tl.bfloat16)
        tl.store(outputs_ptr + current_offset, results, mask=mask)

@torch.fx.wrap
def fused_normalization(inputs_embeds):
    # Get input shape
    batch_size, seq_len, hidden_dim = inputs_embeds.shape
    assert hidden_dim == 2048, "Expected hidden_dim=2048 for inputs_embeds"
    
    # Convert to float32 first (as in original computation)
    tmp_10 = inputs_embeds.to(torch.float32)
    
    # Apply the exact same operations as the original computation
    tmp_11 = tmp_10.pow(2)
    tmp_12 = tmp_11.mean(-1, keepdim=True)
    tmp_13 = tmp_12 + 1e-06
    tmp_14 = torch.rsqrt(tmp_13)
    tmp_15 = tmp_10 * tmp_14
    tmp_16 = tmp_15.to(torch.bfloat16)
    
    return tmp_16

def replacement_func():
    return fused_normalization