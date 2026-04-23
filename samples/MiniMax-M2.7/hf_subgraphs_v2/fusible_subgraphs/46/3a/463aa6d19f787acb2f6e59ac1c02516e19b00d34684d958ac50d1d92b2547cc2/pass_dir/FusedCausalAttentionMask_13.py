import torch
import triton
import triton.language as tl

# Pattern for seq_len=13
def pattern(in_0):
    # Step 1: Create causal mask (upper triangular with -inf)
    tmp_1 = torch.arange(0, 13, device=device(type='cuda', index=0))
    tmp_2 = torch.full((13, 13), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=device(type='cuda', index=0))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(13, device=device(type='cuda', index=0))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    
    # Step 2: Expand and clone to 4D
    tmp_8 = tmp_7[None, None, slice(None, None, None), slice(None, None, None)]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    
    # Step 3: Add input mask and apply zero-masking
    tmp_11 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)]
    tmp_12 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    tmp_13 = tmp_12.to(device(type='cuda', index=0))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_10[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 13, None)] = tmp_17
    
    # Step 4: Remove all-inf rows
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return tmp_22


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_attention_mask_kernel_13(
    input_ptr,
    output_ptr,
    stride_input: int,
    n_elements: int,
    BLOCK_SIZE: tl.constexpr,
):
    seq_len = 13
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    total_per_batch = seq_len * seq_len
    batch_idx = offsets // total_per_batch
    remainder = offsets % total_per_batch
    row_idx = remainder // seq_len
    col_idx = remainder % seq_len
    
    causal_mask = tl.where(row_idx > col_idx, float('-inf'), 0.0)
    
    batch_offset = batch_idx * stride_input
    input_offset = batch_offset + row_idx * stride_input
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    input_val = input_val.to(tl.float32)
    
    val = causal_mask + input_val
    val = tl.where(val == 0.0, float('-inf'), val)
    
    tl.store(output_ptr + offsets, val, mask=mask)


@triton.jit
def filter_rows_kernel_13(
    output_ptr,
    num_rows: int,
    BLOCK_SIZE: tl.constexpr,
):
    seq_len = 13
    pid = tl.program_id(0)
    if pid >= num_rows:
        return
    
    row_start = pid * seq_len
    
    all_neg_inf = True
    for i in range(13):
        val = tl.load(output_ptr + row_start + i)
        if val != float('-inf'):
            all_neg_inf = False
            break
    
    if all_neg_inf:
        for i in range(13):
            tl.store(output_ptr + row_start + i, 0.0)


@torch.fx.wrap
def fused_attention_mask_13(input_tensor):
    batch_size, seq_len = input_tensor.shape
    output_shape = (batch_size, 1, seq_len, seq_len)
    
    if input_tensor.device.type != 'cuda':
        input_tensor = input_tensor.to('cuda')
    
    output = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    n_elements = output.numel()
    stride_input = input_tensor.stride(0)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_attention_mask_kernel_13[(num_programs,)](
        input_tensor,
        output,
        stride_input,
        n_elements,
        BLOCK_SIZE,
    )
    
    num_rows = batch_size * 1 * seq_len
    
    filter_rows_kernel_13[(num_rows,)](
        output,
        num_rows,
        1,
    )
    
    return output


def replacement_func():
    return fused_attention_mask_13