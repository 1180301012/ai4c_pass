import torch
import triton
import triton.language as tl

def pattern(attention_mask, input_ids):
    """
    Pattern: Redundant padding detection (tmp_8 = tmp_1.__eq__(2)) 
    that's computed after the first padding detection (tmp_4 = tmp_1.__eq__(2))
    """
    # This matches the second redundant padding detection
    padding_mask2 = input_ids.__eq__(2)
    return padding_mask2


def replacement_args(attention_mask, input_ids):
    return (attention_mask, input_ids)


@triton.jit
def count_padding_tokens_kernel(
    input_ids_ptr,
    padding_counts_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Each program handles one sequence
    seq_start = pid * seq_len
    seq_end = seq_start + seq_len
    
    total_padding = 0
    for i in range(seq_len):
        idx = seq_start + i
        if idx < seq_len * batch_size:  # Safety check
            token_id = tl.load(input_ids_ptr + idx)
            if token_id == 2:
                total_padding += 1
    
    tl.store(padding_counts_ptr + pid, total_padding)


@triton.jit  
def create_padding_mask_kernel(
    input_ids_ptr,
    padding_mask_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size * seq_len:
        return
    
    token_id = tl.load(input_ids_ptr + pid)
    is_padding = token_id == 2
    
    tl.store(padding_mask_ptr + pid, is_padding)


@torch.fx.wrap
def optimized_padding_operations(attention_mask, input_ids):
    batch_size, seq_len = input_ids.shape
    
    # Single kernel to create all padding masks
    padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=input_ids.device)
    
    total_elements = batch_size * seq_len
    BLOCK_SIZE = 256
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    create_padding_mask_kernel[(num_programs,)](
        input_ids_ptr=input_ids,
        padding_mask_ptr=padding_mask,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Expand mask for dimension matching
    mask_expanded = padding_mask.unsqueeze(-1)
    
    # Count padding tokens per sequence
    padding_counts = torch.zeros(batch_size, dtype=torch.int32, device=input_ids.device)
    
    num_batch_programs = (batch_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    count_padding_tokens_kernel[(num_batch_programs,)](
        input_ids_ptr=input_ids,
        padding_counts_ptr=padding_counts,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Convert to float for division operations in the main computation
    padding_counts_float = padding_counts.float()
    
    return padding_mask, mask_expanded, padding_counts_float


def replacement_func():
    return optimized_padding_operations