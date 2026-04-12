import torch
import triton
import triton.language as tl

def pattern(tmp_1, tmp_0):
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    return tmp_5

def replacement_args(tmp_1, tmp_0):
    return (tmp_1, tmp_0)

@triton.jit
def fused_normalization_kernel_single(
    weighted_ptr,
    weight_ptr,
    out_ptr,
    batch_size,
    seq_len,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * hidden_size:
        return
    
    # Calculate positions
    hidden_idx = pid % hidden_size
    batch_idx = pid // hidden_size
    
    # Compute weighted sum and weight sum in a single pass
    weighted_sum = 0.0
    weight_sum = 0.0
    
    for seq_idx in range(0, seq_len):
        # Load weighted value
        weighted_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx
        weighted_val = tl.load(weighted_ptr + weighted_offset)
        weighted_sum += weighted_val
        
        # Load weight value
        weight_offset = batch_idx * seq_len * hidden_size + seq_idx * hidden_size + hidden_idx
        weight_val = tl.load(weight_ptr + weight_offset)
        weight_sum += weight_val
    
    # Compute fused normalization: weighted_sum / clamp(weight_sum, min_val)
    clamped_weight = tl.maximum(weight_sum, 1e-09)
    result = weighted_sum / clamped_weight
    
    # Store result
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def fused_normalization_wrapper(weighted_tensor, weight_tensor):
    batch_size, seq_len, hidden_size = weight_tensor.shape
    
    out = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=weight_tensor.device)
    
    BLOCK_SIZE = 1024
    grid_size = (batch_size * hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Compute fused normalization in a single kernel
    fused_normalization_kernel_single[(grid_size,)](
        weighted_tensor,
        weight_tensor,
        out,
        batch_size,
        seq_len,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_normalization_wrapper