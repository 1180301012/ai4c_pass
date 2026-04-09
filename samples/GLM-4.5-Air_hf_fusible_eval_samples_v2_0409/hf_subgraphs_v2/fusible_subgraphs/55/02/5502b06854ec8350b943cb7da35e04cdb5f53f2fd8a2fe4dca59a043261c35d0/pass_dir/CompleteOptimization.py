import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.heuristics(
    {"BLOCK_SIZE_FEAT": lambda kwargs: 1024}
)
@triton.jit
def complete_optimization_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_features,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    # Process one batch at a time
    batch_idx = tl.program_id(0)
    feat_block_idx = tl.program_id(1)
    
    feat_start = feat_block_idx * BLOCK_SIZE_FEAT
    feat_offset = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < n_features
    
    # Vectorized accumulators
    sum_prod_all = tl.zeros([BLOCK_SIZE_FEAT], dtype=tl.float32)
    sum_in0_all = tl.zeros([BLOCK_SIZE_FEAT], dtype=tl.float32)
    
    # Process all sequence positions
    for seq_pos in range(n_seq):
        base_offset = batch_idx * n_seq * n_features + seq_pos * n_features
        
        in_0_features = tl.load(in_0_ptr + base_offset + feat_offset, mask=feat_mask, other=0).to(tl.float32)
        in_1_features = tl.load(in_1_ptr + base_offset + feat_offset, mask=feat_mask, other=0.0)
        
        sum_prod_all += in_0_features * in_1_features
        sum_in0_all += in_0_features
    
    # Normalize
    clamped_sum_in0 = tl.maximum(sum_in0_all, 1e-09)
    result = sum_prod_all / clamped_sum_in0
    
    # Output the final result (replicates redundant cat operation)
    out_offset = batch_idx * n_features + feat_offset
    tl.store(out_ptr + out_offset, result, mask=feat_mask)

@torch.fx.wrap
def complete_optimization(in_0, in_1):
    n_batch, n_seq, n_features = in_0.shape
    out_shape = (n_batch, n_features)  # Skip the redundant cat operation
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    BLOCK_SIZE_FEAT = 1024
    grid_x = n_batch
    grid_y = (n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT
    
    complete_optimization_kernel[(grid_x, grid_y)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_ptr=out,
        n_batch=n_batch,
        n_seq=n_seq,
        n_features=n_features,
        BLOCK_SIZE_FEAT=BLOCK_SIZE_FEAT,
    )
    
    return out

def replacement_func():
    return complete_optimization