import torch
import triton
import triton.language as tl

# Define the pattern to match - this matches the causal attention mask computation
def pattern(in_0):
    # Step 1: Create causal mask (upper triangular with -inf)
    seq_len = 21  # Will be extracted via replacement_args
    tmp_1 = torch.arange(0, seq_len, device=torch.device('cuda'))
    tmp_2 = torch.full((seq_len, seq_len), fill_value=-3.4028234663852886e+38, dtype=torch.float32, device=torch.device('cuda'))
    tmp_3 = torch.triu(tmp_2, diagonal=1)
    tmp_4 = torch.arange(seq_len, device=torch.device('cuda'))
    tmp_5 = tmp_1.reshape(-1, 1)
    tmp_6 = tmp_4 > tmp_5
    tmp_3 *= tmp_6
    tmp_7 = tmp_3
    
    # Step 2: Expand and clone to 4D
    tmp_8 = tmp_7[(None, None, slice(None, None, None), slice(None, None, None))]
    tmp_9 = tmp_8.expand(1, 1, -1, -1)
    tmp_10 = tmp_9.clone()
    
    # Step 3: Add input mask and apply zero-masking
    tmp_11 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, seq_len, None))]
    tmp_12 = in_0[(slice(None, None, None), None, None, slice(None, None, None))]
    tmp_13 = tmp_12.to(torch.device('cuda'))
    tmp_14 = tmp_11 + tmp_13
    tmp_15 = tmp_14.__eq__(0)
    tmp_16 = tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, seq_len, None))]
    tmp_17 = tmp_16.masked_fill(tmp_15, -3.4028234663852886e+38)
    tmp_10[(slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, seq_len, None))] = tmp_17
    
    # Step 4: Remove all-inf rows
    tmp_19 = tmp_10.__eq__(-3.4028234663852886e+38)
    tmp_20 = torch.all(tmp_19, dim=-1, keepdim=True)
    tmp_21 = ~tmp_20
    tmp_22 = tmp_10.mul(tmp_21)
    return tmp_22


@triton.jit
def fused_attention_mask_kernel(
    input_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    stride_input: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one output element
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input mask [batch, seq_len]
    input_vals = tl.load(input_ptr + offsets * stride_input, mask=mask, other=0.0)
    input_vals = input_vals.to(tl.float32)
    
    # Calculate indices for the 4D output [1, 1, seq_len, seq_len]
    # output_idx = b * seq_len * seq_len + i * seq_len + j
    # We need to reconstruct b from the linear offset
    total_per_batch = seq_len * seq_len
    batch_idx = offsets // total_per_batch
    remainder = offsets % total_per_batch
    row_idx = remainder // seq_len
    col_idx = remainder % seq_len
    
    # Compute causal mask value (-inf if row > col)
    causal_mask = tl.where(row_idx > col_idx, float('-inf'), 0.0)
    
    # Broadcast input to match [1, 1, seq_len, seq_len]
    # input_vals has shape [batch, seq_len], we need to select based on batch and row
    batch_offset = batch_idx * stride_input
    input_offset = batch_offset + row_idx * stride_input
    input_val = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    input_val = input_val.to(tl.float32)
    
    # Add causal mask and input
    val = causal_mask + input_val
    
    # Replace zeros with -inf
    val = tl.where(val == 0.0, float('-inf'), val)
    
    # Store intermediate result to check all-inf rows later
    tl.store(output_ptr + offsets, val, mask=mask)


@triton.jit
def filter_all_inf_rows_kernel(
    output_ptr,
    seq_len: tl.constexpr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Process row by row, setting all-inf rows to 0
    pid = tl.program_id(0)
    row_start = pid * seq_len
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the row
    row_vals = tl.load(output_ptr + offsets, mask=mask, other=float('-inf'))
    
    # Check if all values are -inf
    all_inf = True
    for i in range(tl.constexpr(256)):  # Max seq_len support
        if i >= seq_len:
            break
        if offsets + i < n_elements:
            val = tl.load(output_ptr + row_start + i)
            if val != float('-inf'):
                all_inf = False
                break
    
    # Set row to 0 if all -inf
    if all_inf:
        for i in range(tl.constexpr(256)):
            if i >= seq_len:
                break
            if row_start + i < n_elements:
                tl.store(output_ptr + row_start + i, 0.0)


@torch.fx.wrap
def fused_attention_mask(input_tensor):
    batch_size, seq_len = input_tensor.shape
    output_shape = (batch_size, 1, seq_len, seq_len)
    
    # Move input to cuda if needed
    if input_tensor.device.type != 'cuda':
        input_tensor = input_tensor.to('cuda')
    
    # Output tensor
    output = torch.empty(output_shape, dtype=torch.float32, device='cuda')
    n_elements = output.numel()
    stride_input = input_tensor.stride(0)
    
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel to create causal mask and add input
    fused_attention_mask_kernel[(num_programs,)](
        input_tensor,
        output,
        seq_len,
        stride_input,
        n_elements,
        BLOCK_SIZE,
    )
    
    # Second kernel to filter all-inf rows
    # For each row, check if all values are -inf and set to 0 if so
    num_rows = batch_size * 1 * seq_len  # total rows in output
    
    # Actually, let me do this differently - process row by row in a separate kernel
    @triton.jit
    def filter_rows_kernel(
        output_ptr,
        seq_len: tl.constexpr,
        num_rows: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)
        if pid >= num_rows:
            return
        
        row_start = pid * seq_len
        
        # Load all values in this row
        row_vals = tl.load(output_ptr + row_start + tl.arange(0, tl.constexpr(256)))
        
        # Check if all are -inf
        all_neg_inf = True
        for i in range(tl.constexpr(256)):
            if i >= seq_len:
                break
            val = row_vals[i]
            if val != float('-inf'):
                all_neg_inf = False
                break
        
        # If all -inf, zero out the row
        if all_neg_inf:
            for i in range(tl.constexpr(256)):
                if i >= seq_len:
                    break
                tl.store(output_ptr + row_start + i, 0.0)
    
    filter_rows_kernel[(num_rows,)](
        output,
        seq_len,
        num_rows,
        1,  # BLOCK_SIZE
    )
    
    return output


def replacement_args(in_0):
    # Extract the input tensor
    return (in_0,)


def replacement_func():
    return fused_attention_mask