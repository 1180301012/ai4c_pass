import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Match mean operation along dim=-2 with keepdim=True"""
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def mean_dim_n2_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    feat_dim,
    reduce_dim,
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_F: tl.constexpr,
):
    """Optimized mean kernel for dim=-2 with keepdim=True"""
    # Each program handles a batch element and a feature block
    batch_id = tl.program_id(0) * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)
    feat_id = tl.program_id(1) * BLOCK_SIZE_F + tl.arange(0, BLOCK_SIZE_F)
    
    mask_batch = batch_id < batch_size
    mask_feat = feat_id < feat_dim
    mask_complete = mask_batch & mask_feat
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_B, BLOCK_SIZE_F], dtype=tl.float32)
    
    # Load input data and compute sum
    for k in range(reduce_dim):
        in_offset = (batch_id.unsqueeze(1) * reduce_dim + k) * feat_dim + feat_id.unsqueeze(1)
        val = tl.load(in_ptr + in_offset, mask=mask_complete.unsqueeze(1), other=0.0).to(tl.float32)
        acc += val
    
    # Compute mean by dividing by reduce dimension size
    mean_val = acc / reduce_dim
    
    # Store result - shape becomes [batch_size, 1, feat_dim]
    out_offset = (batch_id.unsqueeze(1) * feat_dim + feat_id.unsqueeze(1))
    tl.store(out_ptr + out_offset, mean_val, mask=mask_complete.unsqueeze(1))

@torch.fx.wrap
def optimized_mean_dim_n2(in_2):
    # Get input dimensions
    input_shape = in_2.shape
    batch_size = input_shape[0]
    reduce_dim = input_shape[-2]  # Dimension to reduce (dim=-2)
    feat_dim = input_shape[-1]    # Feature dimension to preserve
    
    # Configure kernel launch parameters
    BLOCK_SIZE_B = 64  # Batch elements per program
    BLOCK_SIZE_F = 256  # Features per program
    
    num_batch = (batch_size + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    num_feat = (feat_dim + BLOCK_SIZE_F - 1) // BLOCK_SIZE_F
    
    # Allocate output tensor with keepdim=True
    output_shape = (batch_size, 1, feat_dim)
    output = torch.empty(output_shape, dtype=torch.float32, device=in_2.device)
    
    # Launch kernel
    mean_dim_n2_kernel[(num_batch, num_feat)](
        in_2,
        output,
        batch_size,
        feat_dim,
        reduce_dim,
        BLOCK_SIZE_B,
        BLOCK_SIZE_F,
    )
    
    return output

def replacement_func():
    return optimized_mean_dim_n2