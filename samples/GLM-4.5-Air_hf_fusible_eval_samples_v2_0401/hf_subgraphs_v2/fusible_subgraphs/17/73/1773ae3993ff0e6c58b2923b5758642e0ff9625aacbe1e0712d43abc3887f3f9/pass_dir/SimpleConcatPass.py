import torch
import triton
import triton.language as tl

def pattern(tensor1, tensor2, tensor3):
    """Pattern: torch.cat((tensor1, tensor2, tensor3), dim=2)"""
    result = torch.cat((tensor1, tensor2, tensor3), dim=2)
    return result

def replacement_args(tensor1, tensor2, tensor3):
    return (tensor1, tensor2, tensor3)

@triton.jit
def simple_concat_kernel(
    ptr1, ptr2, ptr3, out_ptr,
    batch_size, total_heads, total_seq_len, total_hidden,
    seq_len1, seq_len2, seq_len3,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    stride = tl.num_programs(0)
    
    # Each program handles multiple elements in parallel
    idx = pid
    while idx < batch_size * total_heads * total_seq_len * total_hidden:
        # Decode index into individual dimensions
        b = idx // (total_heads * total_seq_len * total_hidden)
        h = (idx // (total_seq_len * total_hidden)) % total_heads
        seq = (idx // total_hidden) % total_seq_len
        hidden = idx % total_hidden
        
        # Determine which tensor this sequence position belongs to
        if seq < seq_len1:
            # Load from first tensor (in_2)
            src_seq_idx = seq
            src_ptr = ptr1 + b * total_heads * seq_len1 * total_hidden + \
                     h * seq_len1 * total_hidden + \
                     src_seq_idx * total_hidden + hidden
        elif seq < seq_len1 + seq_len2:
            # Load from second tensor (in_5)
            src_seq_idx = seq - seq_len1
            src_ptr = ptr2 + b * total_heads * seq_len2 * total_hidden + \
                     h * seq_len2 * total_hidden + \
                     src_seq_idx * total_hidden + hidden
        else:
            # Load from third tensor (in_3)
            src_seq_idx = seq - seq_len1 - seq_len2
            src_ptr = ptr3 + b * total_heads * seq_len3 * total_hidden + \
                     h * seq_len3 * total_hidden + \
                     src_seq_idx * total_hidden + hidden
        
        # Calculate output position
        out_idx = idx
        tl.store(out_ptr + out_idx, tl.load(src_ptr))
        
        # Move to next element
        idx += stride

@torch.fx.wrap
def optimized_concat_simple(tensor1, tensor2, tensor3):
    # Reshape tensors to be contiguous for efficient concatenation
    t1 = tensor1.contiguous()
    t2 = tensor2.contiguous() 
    t3 = tensor3.contiguous()
    
    batch_size, num_heads, seq_len1, hidden_size = t1.shape
    _, _, seq_len2, _ = t2.shape
    _, _, seq_len3, _ = t3.shape
    
    total_seq_len = seq_len1 + seq_len2 + seq_len3
    output_shape = (batch_size, num_heads, total_seq_len, hidden_size)
    output = torch.empty(output_shape, dtype=tensor1.dtype, device=tensor1.device)
    
    # Use 1D grid for simplicity and better compatibility
    BLOCK_SIZE = 1024
    total_elements = batch_size * num_heads * total_seq_len * hidden_size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Grid must be a tuple, even for 1D
    simple_concat_kernel[(num_programs,)](
        t1, t2, t3, output,
        batch_size, num_heads, total_seq_len, hidden_size,
        seq_len1, seq_len2, seq_len3,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_concat_simple