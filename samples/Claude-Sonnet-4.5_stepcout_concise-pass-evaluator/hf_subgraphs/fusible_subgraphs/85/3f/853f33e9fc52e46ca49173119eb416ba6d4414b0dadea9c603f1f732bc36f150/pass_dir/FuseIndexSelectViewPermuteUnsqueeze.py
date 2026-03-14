import torch
import triton
import triton.language as tl

def pattern(in_3, in_4):
    """
    Pattern: index_select + view + permute + contiguous + unsqueeze
    """
    tmp_2 = in_3[in_4]
    tmp_3 = tmp_2.view(197, 197, -1)
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = tmp_5.unsqueeze(0)
    return tmp_6

def replacement_args(in_3, in_4):
    return (in_3, in_4)

@triton.jit
def fused_index_select_reshape_kernel(
    in_ptr,
    idx_ptr,
    out_ptr,
    n_indices,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for index_select + view + permute + unsqueeze
    Input: in_ptr[732, n_channels], idx_ptr[38809]
    Output: out_ptr[1, n_channels, 197, 197]
    """
    # Each program handles BLOCK_SIZE output elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_indices
    
    # Load indices
    indices = tl.load(idx_ptr + offsets, mask=mask, other=0)
    
    # For each output position, compute the channel, row, col
    channel = offsets // (197 * 197)
    remainder = offsets % (197 * 197)
    row = remainder // 197
    col = remainder % 197
    
    # Original indexing: in_3[in_4[i]] where i = row * 197 + col
    input_idx = row * 197 + col
    input_idx_masked = tl.where(mask, input_idx, 0)
    table_idx = tl.load(idx_ptr + input_idx_masked, mask=mask, other=0)
    
    # Load from input table: in_ptr[table_idx, channel]
    in_offset = table_idx * n_channels + channel
    value = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Store to output with permuted layout
    tl.store(out_ptr + offsets, value, mask=mask)

@torch.fx.wrap
def fused_index_select_reshape(in_3, in_4):
    """
    Optimized fusion of index_select + view + permute + contiguous + unsqueeze
    """
    # Get dimensions
    n_table, n_channels = in_3.shape
    n_indices = in_4.shape[0]
    
    # Move indices to GPU if on CPU
    if in_4.device.type == 'cpu':
        in_4_cuda = in_4.cuda()
    else:
        in_4_cuda = in_4
    
    # Output shape after all operations: [1, n_channels, 197, 197]
    output_shape = (1, n_channels, 197, 197)
    out = torch.empty(output_shape, device=in_3.device, dtype=in_3.dtype)
    
    # Total output elements
    total_elements = n_channels * 197 * 197
    
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    
    # Flatten for kernel processing
    out_flat = out.view(-1)
    
    fused_index_select_reshape_kernel[grid](
        in_3,
        in_4_cuda,
        out_flat,
        total_elements,
        n_channels,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_index_select_reshape