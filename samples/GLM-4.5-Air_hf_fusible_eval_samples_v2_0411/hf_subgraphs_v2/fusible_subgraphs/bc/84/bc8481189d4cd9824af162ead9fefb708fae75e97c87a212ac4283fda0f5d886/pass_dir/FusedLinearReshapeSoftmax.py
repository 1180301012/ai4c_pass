import torch
import triton
import triton.language as tl

@triton.jit
def fused_softmax_kernel(
    input_ptr,
    output_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    split_size: tl.constexpr,
    head_size: tl.constexpr,
):
    """Optimized kernel for reshaping [B, S, H*S'] to [B, S*S', H'] and applying softmax across H'"""
    pid = tl.program_id(0)
    
    if pid >= batch_size * seq_len * split_size:
        return
    
    # Calculate batch, sequence, and split indices
    batch_idx = pid // (seq_len * split_size)
    remainder = pid % (seq_len * split_size)
    seq_idx = remainder // split_size
    split_idx = remainder % split_size
    
    # Load values for this batch, sequence position, and split  
    max_val = -float('inf')
    sum_val = 0.0
    
    # Find max along the head_size dimension (which is now split across original features)
    for h in range(head_size):
        orig_feature_idx = split_idx * head_size + h
        input_offset = (batch_idx * seq_len * (split_size * head_size) + 
                       seq_idx * (split_size * head_size) + orig_feature_idx)
        val = tl.load(input_ptr + input_offset, other=0.0)
        max_val = tl.maximum(max_val, val)
    
    # Compute exponentials and sum
    for h in range(head_size):
        orig_feature_idx = split_idx * head_size + h
        input_offset = (batch_idx * seq_len * (split_size * head_size) + 
                       seq_idx * (split_size * head_size) + orig_feature_idx)
        val = tl.load(input_ptr + input_offset, other=0.0)
        exp_val = tl.exp(val - max_val)
        sum_val += exp_val
    
    # Store softmax normalized values
    for h in range(head_size):
        orig_feature_idx = split_idx * head_size + h
        input_offset = (batch_idx * seq_len * (split_size * head_size) + 
                       seq_idx * (split_size * head_size) + orig_feature_idx)
        val = tl.load(input_ptr + input_offset, other=0.0)
        exp_val = tl.exp(val - max_val)
        output_val = exp_val / sum_val
        tl.store(output_ptr + input_offset, output_val)

@triton.jit
def linear_reshape_kernel(
    x_ptr,  # input [B, S, F] = [1, 19, 128]
    w_ptr,  # weights [O, F] = [18, 128] 
    b_ptr,  # bias [O] = [18]
    out_ptr,  # intermediate output [B, S, O] = [1, 19, 18]
    B: tl.constexpr,
    S: tl.constexpr,
    F: tl.constexpr,
    O: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """Linear layer kernel with optimized tiling"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Bounds checking
    if pid_m >= B * S or pid_n >= O:
        return
    
    # Calculate batch and sequence indices
    batch_idx = pid_m // S
    seq_idx = pid_m % S
    output_dim = pid_n
    
    # Compute linear combination for output_dim
    bias_val = tl.load(b_ptr + output_dim)
    acc = bias_val
    
    # Parallel reduction over features
    for k in range(0, F, BLOCK_SIZE_N):
        # Load block of weights
        for bk in range(0, min(BLOCK_SIZE_N, F - k)):
            w_offset = output_dim * F + (k + bk)
            w_val = tl.load(w_ptr + w_offset)
            
            # Load block of input
            x_offset = batch_idx * S * F + seq_idx * F + (k + bk)
            x_val = tl.load(x_ptr + x_offset)
            
            acc += x_val * w_val
    
    # Store result
    out_offset = batch_idx * S * O + seq_idx * O + output_dim
    tl.store(out_ptr + out_offset, acc)


    
    # Step 1: Linear layer using Triton kernel
    linear_out = torch.empty(B * S * O, device=x.device, dtype=x.dtype)
    
    # Launch linear kernel
    grid_m = (B * S + 31) // 32  # Block size 32 for M dimension
    grid_n = (O + 31) // 32      # Block size 32 for N dimension
    grid = (grid_m, grid_n)
    
    linear_reshape_kernel[grid](
        x_ptr=x,
        w_ptr=w,
        b_ptr=b,
        out_ptr=linear_out,
        B=B,
        S=S,
        F=F,
        O=O,
        BLOCK_SIZE_M=32,
        BLOCK_SIZE_N=32,
    )
    
    # Reshape linear output to [B, S, split_size, head_size] 
    reshaped = linear_out.reshape(B, S, split_size, head_size)
    
    # Step 2: Apply softmax using Triton kernel
    # Each program handles one batch sequence position for one split dimension
    total_elements = B * S * split_size
    
    # Launch softmax kernel (we'll use the existing reshape_softmax_kernel for this)
    grid = (total_elements + 255 - 1) // 256,
    
    # Flatten reshaped tensor back to linear for kernel processing
    flat_input = reshaped.flatten()
    softmax_result = torch.empty_like(flat_input)
    
    reshape_softmax_kernel[grid](
        input_ptr=flat_input,
        output_ptr=softmax_result,
        batch_size=B,
        seq_len=S,
        feature_dim=split_size,
        split_size=head_size,
    )
    
    # Final reshape to [S, split_size, head_size] and remove singleton dimensions
    final_result = softmax_result.reshape(B, S, split_size, head_size).squeeze(-1)
    
    return final_result.squeeze(0) if B == 1 else final_result

def pattern(in_0, in_1, in_2):
    """
    Matches the fusion of linear -> reshape -> softmax pattern
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

def replacement_func():
    return fused_linear_reshape_softmax_optimized