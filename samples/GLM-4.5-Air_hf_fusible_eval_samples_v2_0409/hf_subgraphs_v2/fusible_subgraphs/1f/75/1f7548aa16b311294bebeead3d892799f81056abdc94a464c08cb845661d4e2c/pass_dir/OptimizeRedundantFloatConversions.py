import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """Match the redundant float conversions pattern"""
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(3, device=device(type='cuda', index=0))
    tmp_3 += 0
    tmp_4 = tmp_3
    tmp_3 = None
    tmp_5 = tmp_2[(slice(None, None, None), tmp_4)]
    tmp_2 = tmp_4 = None
    tmp_6 = torch.arange(3, device=device(type='cuda', index=0))
    tmp_6 += 0
    tmp_7 = tmp_6
    tmp_6 = None
    tmp_8 = in_2.view(-1, 1)
    tmp_9 = tmp_7 <= tmp_8
    tmp_7 = tmp_8 = None
    tmp_10 = tmp_9[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = None
    tmp_11 = tmp_10.expand(1, -1, -1, -1)
    tmp_10 = None
    tmp_12 = tmp_5[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_5 = None
    tmp_13 = tmp_11 * tmp_12
    tmp_11 = tmp_12 = None
    _set_grad_enabled = torch.set_grad_enabled(False)
    _set_grad_enabled = None
    tmp_15 = in_1[(None, slice(None, None, None), None)]
    tmp_15 = None
    tmp_16 = tmp_15.float()
    tmp_15 = None
    tmp_17 = tmp_16.expand(1, -1, 1)
    tmp_16 = None
    tmp_18 = tmp_17.to(device(type='cuda', index=0))
    tmp_17 = None
    tmp_19 = in_3[(slice(None, None, None), None, slice(None, None, None))]
    tmp_20 = tmp_19.float()
    tmp_19 = None
    tmp_21 = tmp_18.float()
    tmp_18 = None
    tmp_22 = tmp_20.float()
    tmp_20 = None
    return (tmp_13, tmp_21, tmp_22)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_1, in_3)

@triton.jit
def optimized_float_conversion_kernel(
    inv_freq_ptr,
    position_ids_ptr,
    output1_ptr,
    output2_ptr,
    hidden_size,
    seq_len,
    device_idx: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel that eliminates redundant float conversions"""
    # Calculate program ID
    global_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = global_idx < hidden_size
    
    if mask:
        # Load rotary embeddings (inv_freq) - no redundant float conversion needed
        inv_freq_val = tl.load(inv_freq_ptr + global_idx, mask=mask)
        
        # Load position IDs and ensure they're on correct device
        pos_ids_val = tl.load(position_ids_ptr + global_idx, mask=mask)
        
        # Store the results directly without redundant conversions
        # inv_freq is already in correct dtype, just ensure it's on device
        tl.store(output1_ptr + global_idx, inv_freq_val, mask=mask)
        
        # position_ids converted directly to float
        tl.store(output2_ptr + global_idx, pos_ids_val.to(tl.float32), mask=mask)

@torch.fx.wrap
def optimized_float_conversions(inv_freq, position_ids):
    """Optimized function that eliminates redundant float conversions"""
    # Get tensor dimensions
    hidden_size = inv_freq.shape[0]
    seq_len = position_ids.shape[1]
    
    # Ensure inputs are on the correct device
    device = torch.device('cuda:0')
    inv_freq = inv_freq.to(device=device)
    position_ids = position_ids.to(device=device)
    
    # Create output tensors
    output1 = torch.empty_like(inv_freq)
    output2 = torch.empty(position_ids.shape, dtype=torch.float32, device=device)
    
    # Set up kernel launch
    block_size = 1024
    grid_size = (hidden_size + block_size - 1) // block_size
    
    # Launch optimized kernel
    optimized_float_conversion_kernel[grid_size,](
        inv_freq_ptr=inv_freq,
        position_ids_ptr=position_ids,
        output1_ptr=output1,
        output2_ptr=output2,
        hidden_size=hidden_size,
        seq_len=seq_len,
        device_idx=0,
        BLOCK_SIZE=block_size
    )
    
    return output1, output2

def replacement_func():
    return optimized_float_conversions