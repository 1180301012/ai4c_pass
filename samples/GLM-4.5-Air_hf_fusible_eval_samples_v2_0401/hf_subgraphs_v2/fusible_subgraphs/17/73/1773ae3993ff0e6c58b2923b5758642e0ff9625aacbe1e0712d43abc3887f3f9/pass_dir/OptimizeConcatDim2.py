import torch
import triton
import triton.language as tl

def pattern(tensor1, tensor2, tensor3, input_tensor, weight, bias):
    """Pattern: torch.cat((tensor1, tensor2, tensor3), dim=2) and layer_norm(input_tensor, weight, bias)"""
    concat_result = torch.cat((tensor1, tensor2, tensor3), dim=2)
    hidden_size = weight.shape[0]
    ln_result = torch.nn.functional.layer_norm(input_tensor, (hidden_size,), weight, bias, 1e-12)
    return ln_result, concat_result

def replacement_args(tensor1, tensor2, tensor3, input_tensor, weight, bias):
    return (tensor1, tensor2, tensor3, input_tensor, weight, bias)

@triton.jit
def concat_kernel_3way_dim2(
    # Input pointers
    ptr1, ptr2, ptr3,
    # Output pointer  
    out_ptr,
    # Input and output shapes
    batch_size, num_heads, seq_len1, seq_len2, seq_len3, hidden_size,
    # Block sizes
    BLOCK_BATCH: tl.constexpr,
    BLOCK_HEADS: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_heads = tl.program_id(1)
    pid_seq = tl.program_id(2)
    pid_hidden = tl.program_id(3)
    
    # Create offsets for each dimension
    batch_start = pid_batch * BLOCK_BATCH
    heads_start = pid_heads * BLOCK_HEADS
    seq_start = pid_seq * BLOCK_SEQ
    hidden_start = pid_hidden * BLOCK_HIDDEN
    
    # Calculate total output sequence length
    total_seq_len = seq_len1 + seq_len2 + seq_len3
    
    # Process current block
    for b in range(batch_start, batch_start + BLOCK_BATCH):
        if b >= batch_size:
            continue
        for h in range(heads_start, heads_start + BLOCK_HEADS):
            if h >= num_heads:
                continue
            for seq in range(seq_start, seq_start + BLOCK_SEQ):
                if seq >= total_seq_len:
                    continue
                for hidden in range(hidden_start, hidden_start + BLOCK_HIDDEN):
                    if hidden >= hidden_size:
                        continue
                    
                    # Determine which input tensor this position belongs to
                    if seq < seq_len1:
                        # Load from first tensor
                        src_ptr = ptr1 + b * num_heads * seq_len1 * hidden_size + \
                                 h * seq_len1 * hidden_size + \
                                 seq * hidden_size + hidden
                    elif seq < seq_len1 + seq_len2:
                        # Load from second tensor
                        src_ptr = ptr2 + b * num_heads * seq_len2 * hidden_size + \
                                 h * seq_len2 * hidden_size + \
                                 (seq - seq_len1) * hidden_size + hidden
                    else:
                        # Load from third tensor
                        src_ptr = ptr3 + b * num_heads * seq_len3 * hidden_size + \
                                 h * seq_len3 * hidden_size + \
                                 (seq - seq_len1 - seq_len2) * hidden_size + hidden
                    
                    # Calculate output pointer
                    out_idx = b * num_heads * total_seq_len * hidden_size + \
                             h * total_seq_len * hidden_size + \
                             seq * hidden_size + hidden
                    out_addr = out_ptr + out_idx
                    
                    # Load and store
                    val = tl.load(src_ptr, other=0.0)
                    tl.store(out_addr, val)

@torch.fx.wrap
def optimized_combined_op(tensor1, tensor2, tensor3, input_tensor, weight, bias):
    # Optimized concatenation
    batch_size, num_heads, seq_len1, hidden_size = tensor1.shape
    _, _, seq_len2, _ = tensor2.shape
    _, _, seq_len3, _ = tensor3.shape
    
    total_seq_len = seq_len1 + seq_len2 + seq_len3
    concat_output = torch.empty((batch_size, num_heads, total_seq_len, hidden_size), 
                               dtype=tensor1.dtype, device=tensor1.device)
    
    # Grid configuration for concatenation
    BLOCK_BATCH = 4
    BLOCK_HEADS = 8
    BLOCK_SEQ = 256
    BLOCK_HIDDEN = hidden_size
    
    grid = (
        (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH,
        (num_heads + BLOCK_HEADS - 1) // BLOCK_HEADS,
        (total_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ,
        (hidden_size + BLOCK_HIDDEN - 1) // BLOCK_HIDDEN,
    )
    
    concat_kernel_3way_dim2[grid](
        tensor1, tensor2, tensor3,
        concat_output,
        batch_size, num_heads, seq_len1, seq_len2, seq_len3, hidden_size,
        BLOCK_BATCH, BLOCK_HEADS, BLOCK_SEQ, BLOCK_HIDDEN,
    )
    
    # Optimized layer normalization
    ln_output = optimized_layer_norm_final(input_tensor, weight, bias)
    
    return ln_output, concat_output

def ln_parallel_mean(input_ptr, batch_size, seq_len, hidden_size, EPS: tl.constexpr):
    """Parallel computation of mean for LayerNorm"""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    
    sum_val = 0.0
    for i in range(pid, batch_size * seq_len, stride):
        for h in range(hidden_size):
            input_idx = i * hidden_size + h
            val = tl.load(input_ptr + input_idx, other=0.0)
            sum_val += val
    
    return sum_val

def ln_parallel_var(input_ptr, mean, batch_size, seq_len, hidden_size, EPS: tl.constexpr):
    """Parallel computation of variance for LayerNorm"""
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    
    sum_sq = 0.0
    for i in range(pid, batch_size * seq_len, stride):
        for h in range(hidden_size):
            input_idx = i * hidden_size + h
            val = tl.load(input_ptr + input_idx, other=0.0)
            sum_sq += (val - mean) * (val - mean)
    
    return sum_sq

@triton.jit
def layer_norm_final_optimized_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size, seq_len, hidden_size,
    BLOCK_BATCH: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
    BLOCK_HIDDEN: tl.constexpr,
    EPS: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_seq = tl.program_id(1)
    
    # Offset for current (batch, seq) position
    pos_idx = pid_batch * seq_len + pid_seq
    
    if pos_idx >= batch_size * seq_len:
        return
    
    # Each thread block handles one (batch, seq) pair
    sum_local = 0.0
    sum_sq_local = 0.0
    
    # First pass: compute mean
    for h in range(BLOCK_HIDDEN):
        if h < hidden_size:
            input_idx = pos_idx * hidden_size + h
            val = tl.load(input_ptr + input_idx, other=0.0)
            sum_local += val
    
    # Compute mean for this block
    mean_local = sum_local / hidden_size
    
    # Second pass: compute variance
    for h in range(BLOCK_HIDDEN):
        if h < hidden_size:
            input_idx = pos_idx * hidden_size + h
            val = tl.load(input_ptr + input_idx, other=0.0)
            sum_sq_local += (val - mean_local) * (val - mean_local)
    
    # Compute standard deviation for this block
    std_local = tl.sqrt(sum_sq_local / hidden_size + EPS)
    
    # Final pass: apply normalization
    for h in range(BLOCK_HIDDEN):
        if h < hidden_size:
            input_idx = pos_idx * hidden_size + h
            weight_val = tl.load(weight_ptr + h)
            bias_val = tl.load(bias_ptr + h)
            input_val = tl.load(input_ptr + input_idx, other=0.0)
            
            # LayerNorm formula: (x - mean) / std * weight + bias
            normalized = (input_val - mean_local) / std_local
            result = normalized * weight_val + bias_val
            
            output_idx = pos_idx * hidden_size + h
            tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def optimized_layer_norm_final(input_tensor, weight, bias):
    batch_size, seq_len, hidden_size = input_tensor.shape
    
    output = torch.empty_like(input_tensor)
    
    # Use smaller block sizes for better GPU utilization
    BLOCK_BATCH = 8
    BLOCK_SEQ = 128
    BLOCK_HIDDEN = hidden_size
    
    # Grid configuration
    grid_batch = (batch_size + BLOCK_BATCH - 1) // BLOCK_BATCH
    grid_seq = (seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    # Launch the optimized kernel
    layer_norm_final_optimized_kernel[(grid_batch, grid_seq, 1)](
        input_tensor,
        weight,
        bias,
        output,
        batch_size, seq_len, hidden_size,
        BLOCK_BATCH, BLOCK_SEQ, BLOCK_HIDDEN,
        1e-12
    )
    
    return output

def replacement_func():
    return optimized_combined_op