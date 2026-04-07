import torch
import triton
import triton.language as tl

def pattern(in_6, tmp_5, in_4, scalar):
    tmp_6 = in_6 * tmp_5
    tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)
    tmp_8 = scalar * in_4
    tmp_9 = tmp_8 + tmp_7
    tmp_10 = tmp_9.transpose(1, 2)
    return tmp_10

def replacement_args(in_6, tmp_5, in_4, scalar):
    return (in_6, tmp_5, in_4, scalar)

@triton.jit
def fused_elementwise_kernel(
    in6_ptr, tmp5_ptr, in4_ptr,
    out_ptr,
    N, C, H_orig, W_orig,
    H_padded, 
    scalar_val,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    grid_size = tl.cdiv(N * C * H_padded * W_orig, BLOCK_SIZE)
    if pid >= grid_size:
        return
    
    # Calculate offsets for this thread block
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total_elements = offset
    
    # Convert offset to multi-dimensional indices
    w_idx = total_elements % W_orig
    h_idx = (total_elements // W_orig) % H_padded
    c_idx = (total_elements // (W_orig * H_padded)) % C
    b_idx = total_elements // (W_orig * H_padded * C)
    
    # Create mask for bounds checking
    is_padded_row = (h_idx == 0)  # Check if this is the padded row
    valid_idx = w_idx < W_orig
    
    # Load input tensors
    if is_padded_row:
        # For padded row, load zeros from tmp5 and zeros from in4
        val1 = tl.zeros((BLOCK_SIZE,), dtype=tmp5_ptr.type.element_ty)
        val2 = tl.zeros((BLOCK_SIZE,), dtype=in4_ptr.type.element_ty)
    else:
        # Load from actual data (subtract 1 from h_idx to skip padded row)
        tmp6_offset = b_idx * C * H_orig * W_orig + c_idx * (H_orig-1) * W_orig + (h_idx-1) * W_orig + w_idx
        in4_offset = b_idx * C * H_orig * W_orig + c_idx * H_orig * W_orig + h_idx * W_orig + w_idx
        
        val1 = tl.load(tmp5_ptr + tmp6_offset, mask=(w_idx < W_orig) & (h_idx > 0), other=0.0)
        val2 = tl.load(in4_ptr + in4_offset, mask=(w_idx < W_orig) & (h_idx < H_orig), other=0.0)
    
    # fused operations: multiply by in6, add padded row, multiply by scalar, transpose dims 1 and 2
    out_vals = (in_6 * val1) + (val2 if not is_padded_row else 0.0)
    out_vals = out_vals * scalar_val
    
    # For transpose: need to swap H and C dimensions in output
    # Original output is (N, C, H_padded, W_orig), transpose (1,2) becomes (N, H_padded, C, W_orig)
    # So we need to reorder indices: b_idx, h_idx, c_idx, w_idx
    out_idx = b_idx * H_padded * C * W_orig + h_idx * C * W_orig + c_idx * W_orig + w_idx
    
    mask = offset < (N * H_padded * C * W_orig)
    idx_in_block = offset - (pid * BLOCK_SIZE)
    
    tl.store(out_ptr + out_idx, out_vals, mask=(idx_in_block < BLOCK_SIZE) & valid_idx)

@torch.fx.wrap
def fused_elementwise_ops(in_6, tmp_5, in_4, scalar):
    # Get input dimensions
    N, C, H_orig, W_orig = tmp_5.shape
    N2, C2, H2, W2 = in_4.shape
    
    # The pattern: 
    # tmp_6 = in_6 * tmp_5
    # tmp_7 = torch.nn.functional.pad(tmp_6, (0, 0, 1, 0, 0, 0), 'constant', None)  # pad last 2 dims: (W, H) with (0,0) for W, (1,0) for H
    # tmp_8 = scalar * in_4
    # tmp_9 = tmp_8 + tmp_7
    # tmp_10 = tmp_9.transpose(1, 2)  # swap dims 1 and 2: (N, C, H, W) -> (N, H, C, W)
    
    # After padding, H becomes H + 1
    H_padded = H_orig + 1
    
    # Output after transpose: (N, H_padded, C, W_orig)
    N_out = N
    H_out = H_padded  
    C_out = C
    W_out = W_orig
    
    # Create output tensor with transposed dimensions
    out = torch.empty((N_out, H_out, C_out, W_out), dtype=tmp_5.dtype, device=tmp_5.device)
    
    # Launch kernel
    total_elements = N_out * H_out * C_out * W_out
    BLOCK_SIZE = 1024
    
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_elementwise_kernel[grid_size](
        in6_ptr=in_6,
        tmp5_ptr=tmp_5,
        in4_ptr=in_4,
        out_ptr=out,
        N=N, C=C, H_orig=H_orig, W_orig=W_orig,
        H_padded=H_padded,
        scalar_val=scalar,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def replacement_func():
    return fused_elementwise_ops