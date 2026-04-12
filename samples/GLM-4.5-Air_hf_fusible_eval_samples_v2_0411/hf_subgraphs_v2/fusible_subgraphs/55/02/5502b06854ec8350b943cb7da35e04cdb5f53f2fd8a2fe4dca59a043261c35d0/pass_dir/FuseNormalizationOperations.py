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
def efficient_normalization_kernel(
    weighted_sum_ptr,
    weight_sum_ptr,
    out_ptr,
    batch_size,
    hidden_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * hidden_size:
        return
    
    # Load weighted sum (already computed along sequence dimension)
    weighted_sum = tl.load(weighted_sum_ptr + pid)
    
    # Load weight sum
    weight_sum = tl.load(weight_sum_ptr + pid)
    
    # Compute normalization with clamping
    clamped_weight = tl.maximum(weight_sum, 1e-09)
    result = weighted_sum / clamped_weight
    
    # Store result
    tl.store(out_ptr + pid, result)

@torch.fx.wrap
def efficient_normalization(computed_weighted_sum, computed_weight_sum):
    if len(computed_weight_sum.shape) == 3:
        batch_size, seq_len, hidden_size = computed_weight_sum.shape
        # Perform sum along sequence dimension first
        computed_weighted_sum = computed_weighted_sum.sum(1)
        computed_weight_sum = computed_weight_sum.sum(1)
    else:
        batch_size, hidden_size = computed_weight_sum.shape
    
    out = torch.empty((batch_size, hidden_size), dtype=torch.float32, device=computed_weight_sum.device)
    
    BLOCK_SIZE = 1024
    grid_size = (batch_size * hidden_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    efficient_normalization_kernel[(grid_size,)](
        computed_weighted_sum,
        computed_weight_sum,
        out,
        batch_size,
        hidden_size,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return efficient_normalization