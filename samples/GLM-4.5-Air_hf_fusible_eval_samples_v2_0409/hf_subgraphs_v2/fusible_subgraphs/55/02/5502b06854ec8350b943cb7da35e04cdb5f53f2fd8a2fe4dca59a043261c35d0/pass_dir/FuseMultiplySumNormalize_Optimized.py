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
    return tmp_5

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_multiply_sum_normalize_kernel_optimized(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_features,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    # Each program handles one feature position across all batches and sequences
    batch_idx = tl.program_id(0) // (n_seq * ((n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT))
    seq_idx = (tl.program_id(0) % (n_seq * ((n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT))) // ((n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT)
    feat_block_idx = tl.program_id(0) % ((n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT)
    
    feat_start = feat_block_idx * BLOCK_SIZE_FEAT
    feat_offset = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < n_features
    
    # Initialize accumulators
    sum_prod = tl.zeros([], dtype=tl.float32)
    sum_in0 = tl.zeros([], dtype=tl.float32)
    
    # Process all sequence positions for accumulation
    for seq_pos in range(n_seq):
        # Compute memory offsets directly
        offset = batch_idx * n_seq * n_features + seq_pos * n_features + feat_offset
        
        # Load data with type conversion and bounds checking
        in_0_val = tl.load(in_0_ptr + offset, mask=feat_mask, other=0).to(tl.float32)
        in_1_val = tl.load(in_1_ptr + offset, mask=feat_mask, other=0.0)
        
        # Vectorized operations
        sum_prod += tl.sum(in_0_val * in_1_val)
        sum_in0 += tl.sum(in_0_val)
    
    # Final normalization
    clamped_sum_in0 = tl.maximum(sum_in0, 1e-09)
    result = sum_prod / clamped_sum_in0
    
    # Store result for this batch/sequence position (replicate across features)
    out_offset = batch_idx * n_seq * n_features + seq_idx * n_features + feat_offset
    tl.store(out_ptr + out_offset, result, mask=feat_mask)

@triton.heuristics(
    {"BLOCK_SIZE_FEAT": lambda kwargs: 1024}
)
@triton.jit
def fused_multiply_sum_normalize_kernel_autotuned(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    n_batch,
    n_seq,
    n_features,
    BLOCK_SIZE_FEAT: tl.constexpr,
):
    # Optimized 2D grid: (batch, feature_blocks) with internal sequence processing
    batch_idx = tl.program_id(0)
    feat_block_idx = tl.program_id(1)
    
    feat_start = feat_block_idx * BLOCK_SIZE_FEAT
    feat_offset = feat_start + tl.arange(0, BLOCK_SIZE_FEAT)
    feat_mask = feat_offset < n_features
    
    # Vectorized accumulators for all sequence positions
    sum_prod_all = tl.zeros([BLOCK_SIZE_FEAT], dtype=tl.float32)
    sum_in0_all = tl.zeros([BLOCK_SIZE_FEAT], dtype=tl.float32)
    
    # Parallel processing across sequences
    for seq_pos in range(n_seq):
        # Direct memory access pattern for better coalescing
        base_offset = batch_idx * n_seq * n_features + seq_pos * n_features
        
        in_0_features = tl.load(in_0_ptr + base_offset + feat_offset, mask=feat_mask, other=0).to(tl.float32)
        in_1_features = tl.load(in_1_ptr + base_offset + feat_offset, mask=feat_mask, other=0.0)
        
        # Vectorized accumulation - maximize memory bandwidth usage
        sum_prod_all += in_0_features * in_1_features
        sum_in0_all += in_0_features
    
    # Vectorized normalization
    clamped_sum_in0 = tl.maximum(sum_in0_all, 1e-09)
    result = sum_prod_all / clamped_sum_in0
    
    # Output all sequence positions for this feature block
    for seq_pos in range(n_seq):
        out_base = batch_idx * n_seq * n_features + seq_pos * n_features
        tl.store(out_ptr + out_base + feat_offset, result, mask=feat_mask)

@torch.fx.wrap
def fused_multiply_sum_normalize_optimized(in_0, in_1):
    n_batch, n_seq, n_features = in_0.shape
    out_shape = (n_batch, n_seq, n_features)
    out = torch.empty(out_shape, dtype=torch.float32, device=in_0.device)
    
    # Use heuristic block size selected by Triton
    BLOCK_SIZE_FEAT = 1024
    
    # Optimized 2D grid
    grid_x = n_batch
    grid_y = (n_features + BLOCK_SIZE_FEAT - 1) // BLOCK_SIZE_FEAT
    
    # Use the autotuned version with performance tuning
    fused_multiply_sum_normalize_kernel_autotuned[(grid_x, grid_y)](
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
    return fused_multiply_sum_normalize_optimized