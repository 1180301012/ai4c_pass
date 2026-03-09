import torch
import triton
import triton.language as tl

def pattern(tensor):
    tmp_2 = tensor
    tmp_3 = tmp_2[slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_4 = tmp_3.reshape(1, 12, 12, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    return tmp_6

def replacement_args(tensor):
    return (tensor,)

@triton.jit
def tensor_manipulation_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_rows_original,
    n_cols,
    BLOCK_SIZE_M: tl.constexpr,
):
    """ fused kernel that performs slice(1:) + reshape + permute + contiguous"""
    pid = tl.program_id(0)
    
    # Each program handles one batch element
    if pid >= n_batch:
        return
        
    # Output tensor has shape [n_batch, n_cols, 12, 12]
    # We need to map to this layout directly
    
    # Calculate output grid dimensions
    n_elements_per_col = 12 * 12  # 144
    
    # For each column in original tensor (original had 145 rows, now 144)
    for col_idx in range(0, n_cols, 1):
        col_base = pid * n_rows_original * n_cols + (col_idx + 1) * n_rows_original
        
        # For each position in the 12x12 output grid
        for i in range(12):
            for j in range(12):
                src_offset = col_base + i * 12 + j
                dst_offset = pid * (n_cols * 12 * 12) + col_idx * (12 * 12) + i * 12 + j
                
                mask = src_offset < (pid + 1) * n_rows_original * n_cols
                if mask:
                    val = tl.load(input_ptr + src_offset, mask=mask, other=0.0)
                    tl.store(output_ptr + dst_offset, val, mask=mask)

@torch.fx.wrap
def fused_tensor_operations(tensor):
    n_batch, n_rows_full, n_cols = tensor.shape
    
    # After slice(1:), we have 144 rows
    n_rows_output = 144
    
    # Output shape after reshape + permute: [n_batch, n_cols, 12, 12]
    output_shape = (n_batch, n_cols, 12, 12)
    output = torch.empty(output_shape, dtype=tensor.dtype, device=tensor.device)
    
    # Triton kernel configuration
    BLOCK_SIZE_M = 32
    grid = (n_batch,)
    
    tensor_manipulation_kernel[grid](
        input_ptr=tensor,
        output_ptr=output,
        n_batch=n_batch,
        n_rows_original=n_rows_full,
        n_cols=n_cols,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
    )
    
    return output

def replacement_func():
    return fused_tensor_operations