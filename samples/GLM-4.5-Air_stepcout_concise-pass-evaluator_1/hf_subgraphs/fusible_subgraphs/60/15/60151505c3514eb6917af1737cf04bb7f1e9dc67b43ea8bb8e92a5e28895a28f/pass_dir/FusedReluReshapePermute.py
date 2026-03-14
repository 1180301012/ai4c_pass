import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Match the exact computation pattern from the model
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0.reshape(-1, 256, -1)  # Match original pattern - batch size determined dynamically
    tmp_2 = tmp_0.reshape(-1, 256, -1)
    tmp_3 = tmp_2.permute(0, 2, 1)
    return tmp_3, tmp_1

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_relu_reshape_permute_kernel(
    in_0_ptr,
    in_1_ptr,
    out_0_ptr,  # permuted ReLU result: [batch, seq_len, feature_dim]  
    out_1_ptr,  # reshaped in_0 result: [batch, feature_dim, seq_len]
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    feature_dim: tl.constexpr,
):
    # Get program IDs for 3D grid
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    pid_feat = tl.program_id(2)
    
    # Calculate input offset: [batch, feature_dim, seq_len, 1]
    # Last dimension is 1, so we have [batch, feature_dim, seq_len] effectively
    input_offset = pid_batch * feature_dim * seq_len + pid_feat * seq_len + pid_seq
    
    # Calculate output offsets
    # Output 0: [batch, seq_len, feature_dim] after permute(0, 2, 1)
    offset_0 = pid_batch * seq_len * feature_dim + pid_seq * feature_dim + pid_feat
    
    # Output 1: [batch, feature_dim, seq_len] after reshape
    offset_1 = pid_batch * feature_dim * seq_len + pid_feat * seq_len + pid_seq
    
    # Create mask to stay within bounds
    mask = (pid_batch < batch_size) & (pid_feat < feature_dim) & (pid_seq < seq_len)
    
    if mask:
        # Load from input_1 and apply ReLU
        x = tl.load(in_1_ptr + input_offset, mask=mask, other=0.0)
        relu_out = tl.max(x, 0.0)
        
        # Load from input_0
        x_0 = tl.load(in_0_ptr + input_offset, mask=mask, other=0.0)
        
        # Store results to both outputs
        tl.store(out_0_ptr + offset_0, relu_out)
        tl.store(out_1_ptr + offset_1, x_0)

@torch.fx.wrap
def fused_kernel(in_0, in_1):
    # Get input shapes
    batch_size = in_0.shape[0]
    feature_dim = in_0.shape[1]  # Should be 256
    seq_len = in_0.shape[2]      # Original seq_len (21 or 19)
    seq_len_total = seq_len * in_0.shape[3]  # Total seq length (21*1 or 19*1)
    
    # Create output tensors with correct shapes
    # output_0: [batch, seq_len, feature_dim] after permute(0, 2, 1)
    output_0 = torch.empty((batch_size, seq_len_total, feature_dim), dtype=in_0.dtype, device=in_0.device)
    # output_1: [batch, feature_dim, seq_len_total] after reshape
    output_1 = torch.empty((batch_size, feature_dim, seq_len_total), dtype=in_0.dtype, device=in_0.device)
    
    # Calculate grid size for 3D launch
    num_programs_batch = batch_size
    num_programs_seq = seq_len_total  
    num_programs_feat = feature_dim
    
    grid = (
        num_programs_batch,
        num_programs_seq,
        num_programs_feat
    )
    
    # Launch the kernel
    fused_relu_reshape_permute_kernel[grid](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        out_0_ptr=output_0,
        out_1_ptr=output_1,
        batch_size=batch_size,
        seq_len=seq_len_total,  # Total sequence length after flattening last dimension
        feature_dim=feature_dim,
    )
    
    return output_0, output_1

def replacement_func():
    return fused_kernel