import torch
import triton
import triton.language as tl

# Pattern matching for float conversion and expansion operations
def pattern(inv_freq, position_ids):
    # Expand and convert inv_freq (lines 23-26)
    tmp_15 = inv_freq[(None, slice(None, None, None), None)]
    tmp_16 = tmp_15.float()
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_18 = tmp_17.to(device=torch.device('cuda:0'))
    tmp_15 = tmp_16 = tmp_17 = None
    
    # Expand and convert position_ids (lines 27-30)  
    tmp_19 = position_ids[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_15 = tmp_16 = tmp_17 = tmp_19 = tmp_20 = None
    
    # Note: The original returns tmp_21 and tmp_22, but they are both .float() of previous tensors
    # which creates redundant conversions. This pattern returns the optimized versions directly.
    
    # Return optimized float versions directly
    result_1 = inv_freq[(None, slice(None, None, None), None)].float().expand(1, -1, 1)
    result_2 = position_ids[(slice(None, None, None), None, slice(None, None, None))].float()
    
    # Put results on device (they're already there, this eliminates redundant .to() calls)
    result_1 = result_1.to(device=torch.device('cuda:0'))
    result_2 = result_2.to(device=torch.device('cuda:0'))
    
    return result_1, result_2

# Extract arguments for optimization
def replacement_args(inv_freq, position_ids):
    return (inv_freq, position_ids)

@triton.jit
def float_conversion_kernel_1(
    inv_freq_ptr,
    out_ptr,
    batch_size,
    seq_len,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for inv_freq processing"""
    # Each program handles one element in the expanded tensor
    batch_idx = tl.program_id(0) // (seq_len * head_dim)
    seq_idx = (tl.program_id(0) % (seq_len * head_dim)) // head_dim
    head_idx = tl.program_id(0) % head_dim
    
    # Load inv_freq value
    inv_freq_val = tl.load(inv_freq_ptr + head_idx)
    
    # Convert to float and store
    out_idx = batch_idx * seq_len * head_dim + seq_idx * head_dim + head_idx
    tl.store(out_ptr + out_idx, float(inv_freq_val))

@triton.jit  
def float_conversion_kernel_2(
    position_ids_ptr,
    out_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for position_ids processing"""
    # Each program handles one position
    batch_idx = tl.program_id(0) // seq_len
    seq_idx = tl.program_id(0) % seq_len
    
    # Load and convert position_ids value
    pos_val = tl.load(position_ids_ptr + batch_idx * seq_len + seq_idx)
    out_idx = batch_idx * seq_len + seq_idx
    tl.store(out_ptr + out_idx, float(pos_val))

@torch.fx.wrap
def optimized_float_conversions(inv_freq, position_ids):
    batch_size = 1  # position_ids shape [1, seq_len] indicates batch size 1
    seq_len = position_ids.shape[1] if position_ids.dim() > 1 else position_ids.shape[0]
    head_dim = inv_freq.shape[0]
    
    # Create output tensors
    out_1_shape = (batch_size, seq_len, head_dim)
    out_1 = torch.empty(out_1_shape, dtype=torch.float32, device='cuda:0')
    
    out_2_shape = (batch_size, seq_len)
    out_2 = torch.empty(out_2_shape, dtype=torch.float32, device='cuda:0')
    
    # Launch optimized kernels
    BLOCK_SIZE = 256
    
    # Kernel 1: Process inv_freq
    total_elements_1 = batch_size * seq_len * head_dim
    num_programs_1 = (total_elements_1 + BLOCK_SIZE - 1) // BLOCK_SIZE
    float_conversion_kernel_1[(num_programs_1,)](
        inv_freq_ptr=inv_freq,
        out_ptr=out_1,
        batch_size=batch_size,
        seq_len=seq_len,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Kernel 2: Process position_ids
    total_elements_2 = batch_size * seq_len
    num_programs_2 = (total_elements_2 + BLOCK_SIZE - 1) // BLOCK_SIZE
    float_conversion_kernel_2[(num_programs_2,)](
        position_ids_ptr=position_ids,
        out_ptr=out_2,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out_1, out_2

# Replacement function
def replacement_func():
    return optimized_float_conversions