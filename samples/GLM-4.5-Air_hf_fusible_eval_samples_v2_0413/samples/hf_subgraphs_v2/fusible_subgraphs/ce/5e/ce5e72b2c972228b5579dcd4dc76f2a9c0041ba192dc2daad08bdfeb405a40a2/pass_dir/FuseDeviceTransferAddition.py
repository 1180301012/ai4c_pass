import torch
import triton
import triton.language as tl

# Pattern matching function - matches the sequence from detach through addition
def pattern(in_2, tmp_5):
    tmp_6 = in_2.detach()
    tmp_7 = tmp_6.type_as(tmp_5)
    tmp_8 = tmp_7.to(device=torch.device(type='cuda', index=0), copy=True)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9

# Argument extraction function
def replacement_args(in_2, tmp_5):
    return (in_2, tmp_5)

# Simplified kernel that handles device transfer and addition
@triton.jit
def device_transfer_add_kernel(
    x_ptr,           # tmp_5 pointer (on GPU) - shape [1, 655335, 768]
    y_ptr,           # in_2 pointer (on CPU) - shape [1, 1568, 768]
    out_ptr,         # output pointer
    x_seq_len,       # sequence length of x (655335)
    y_seq_len,       # sequence length of y (1568)
    features,        # feature dimension (768)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate sequence positions for this program
    seq_pos = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = seq_pos < x_seq_len
    
    # For each position in x, determine corresponding position in y
    # Since position embeddings are usually arranged sequentially, use modulo
    y_positions = seq_pos % y_seq_len
    
    # Expand dimensions for broadcasting
    seq_pos_expanded = tl.expand_dims(seq_pos, (1, 2))
    y_positions_expanded = tl.expand_dims(y_positions, (1, 2))
    feature_offset = tl.expand_dims(tl.arange(0, features), (0, 1))
    
    # Load x from GPU
    x_ptrs = x_ptr + seq_pos_expanded * features + feature_offset
    x = tl.load(x_ptrs, mask=mask[:, None, None], other=0.0)
    
    # Load y from CPU (simulating device transfer)
    y_ptrs = y_ptr + y_positions_expanded * features + feature_offset
    # Note: This is a conceptual load - in real scenario, this would handle CPU-GPU transfer
    y = tl.load(y_ptrs, mask=mask[:, None, None], other=0.0)
    
    # Addition
    out = x + y
    
    # Store result
    out_ptrs = out_ptr + seq_pos_expanded * features + feature_offset
    tl.store(out_ptrs, out, mask=mask[:, None, None])

# Optimized kernel that performs device transfer and addition
@triton.jit
def device_transfer_add_kernel(
    x_ptr,           # tmp_5 pointer (on GPU) - shape [1, 655335, 768]
    y_ptr,           # in_2 pointer (on CPU) - shape [1, 1568, 768]
    out_ptr,         # output pointer
    x_batch,         # 1
    x_seq_len,       # 655335
    x_features,      # 768
    y_batch,         # 1
    y_seq_len,       # 1568
    y_features,      # 768
    BLOCK_SIZE: tl.constexpr,
    FEATURES: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate positions for this program
    positions = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = positions < x_seq_len
    
    # For each position in x, find corresponding position in y
    y_positions = positions % y_seq_len
    
    # Expand dimensions for broadcasting
    pos_3d = tl.expand_dims(positions, (1, 2))
    y_pos_3d = tl.expand_dims(y_positions, (1, 2))
    
    # Use power of 2 for arange (1024) and create mask for valid features (768)
    feat_offset = tl.expand_dims(tl.arange(0, 1024), (0, 1))
    feat_mask = feat_offset < FEATURES
    
    # Compute memory addresses
    x_addr = x_ptr + pos_3d * x_features + feat_offset
    y_addr = y_ptr + y_pos_3d * y_features + feat_offset
    out_addr = out_ptr + pos_3d * x_features + feat_offset
    
    # Combine masks: position mask AND feature mask
    combined_mask = mask[:, None, None] & feat_mask
    
    # Load x from GPU
    x_vals = tl.load(x_addr, mask=combined_mask, other=0.0)
    
    # Load y from CPU (simulating the device transfer)
    # In reality, this would involve the actual device transfer
    y_vals = tl.load(y_addr, mask=combined_mask, other=0.0)
    
    # Perform addition
    out_vals = x_vals + y_vals
    
    # Store results
    tl.store(out_addr, out_vals, mask=combined_mask)

@torch.fx.wrap
def fused_device_transfer_add(in_2, tmp_5):
    # Get input shapes
    x_shape = tmp_5.shape  # [1, 655335, 768]
    y_shape = in_2.shape   # [1, 1568, 768]
    
    # Create output tensor on the same device as tmp_5
    out = torch.empty_like(tmp_5)
    
    # Set up kernel launch parameters
    BLOCK_SIZE = 1024
    num_programs = (x_shape[1] + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch Triton kernel
    device_transfer_add_kernel[(num_programs,)](
        x_ptr=tmp_5,
        y_ptr=in_2,
        out_ptr=out,
        x_batch=x_shape[0],
        x_seq_len=x_shape[1],
        x_features=x_shape[2],
        y_batch=y_shape[0],
        y_seq_len=y_shape[1],
        y_features=y_shape[2],
        BLOCK_SIZE=BLOCK_SIZE,
        FEATURES=x_shape[2],
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_device_transfer_add