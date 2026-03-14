import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_2 = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = tmp_2[0]
    tmp_4 = tmp_2[1]
    tmp_5 = tmp_2[2]
    tmp_6 = tmp_5.unsqueeze(2)
    tmp_7 = in_0[None, None, slice(None, None, None)]
    return (tmp_7, tmp_3, tmp_6, tmp_4)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def silu_kernel(x):
    return x * tl.sigmoid(x)

@triton.jit
def optimized_silu_split_unsqueeze_kernel(
    in_1_ptr,
    out_3_ptr,
    out_4_ptr, 
    out_6_ptr,
    batch_size,
    seq_len,
    BLOCK_SIZE_512: tl.constexpr,
    BLOCK_SIZE_128: tl.constexpr,
):
    # Get program IDs for parallel processing
    batch_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Calculate base offsets
    in_1_base = batch_id * seq_len * 1152 + seq_id * 1152
    
    # Process first chunk (512 elements)
    offsets_512 = tl.arange(0, BLOCK_SIZE_512)
    mask_512 = offsets_512 < 512
    in_1_512 = tl.load(in_1_ptr + in_1_base + offsets_512, mask=mask_512)
    out_3_data = silu_kernel(in_1_512)
    tl.store(out_3_ptr + (batch_id * seq_len * 512 + seq_id * 512 + offsets_512), out_3_data, mask=mask_512)
    
    # Process second chunk (512 elements)
    offsets_512_2 = tl.arange(0, BLOCK_SIZE_512)
    mask_512_2 = offsets_512_2 < 512
    in_1_512_2 = tl.load(in_1_ptr + in_1_base + 512 + offsets_512_2, mask=mask_512_2)
    out_4_data = silu_kernel(in_1_512_2)
    tl.store(out_4_ptr + (batch_id * seq_len * 512 + seq_id * 512 + offsets_512_2), out_4_data, mask=mask_512_2)
    
    # Process third chunk (128 elements) with unsqueeze
    offsets_128 = tl.arange(0, BLOCK_SIZE_128)
    mask_128 = offsets_128 < 128
    in_1_128 = tl.load(in_1_ptr + in_1_base + 1024 + offsets_128, mask=mask_128)
    out_6_data = silu_kernel(in_1_128)
    tl.store(out_6_ptr + (batch_id * seq_len * 128 * 3 + seq_id * 128 * 3 + offsets_128 * 3), out_6_data, mask=mask_128)
    
    # Handle padding for unsqueeze dimension
    tl.store(out_6_ptr + (batch_id * seq_len * 128 * 3 + seq_id * 128 * 3 + offsets_128 * 3 + 1), 0.0, mask=mask_128)
    tl.store(out_6_ptr + (batch_id * seq_len * 128 * 3 + seq_id * 128 * 3 + offsets_128 * 3 + 2), 0.0, mask=mask_128)

@torch.fx.wrap
def optimized_silu_split_unsqueeze(in_0, in_1):
    batch_size, seq_len = in_1.shape[0], in_1.shape[1]
    
    # Output shapes
    out_3_shape = (batch_size, seq_len, 512)
    out_4_shape = (batch_size, seq_len, 512)
    out_6_shape = (batch_size, seq_len, 128, 3)  # unsqueeze on dim=2
    
    # Create output tensors
    out_3 = torch.empty(out_3_shape, dtype=in_1.dtype, device=in_1.device)
    out_4 = torch.empty(out_4_shape, dtype=in_1.dtype, device=in_1.device)
    out_6 = torch.empty(out_6_shape, dtype=in_1.dtype, device=in_1.device)
    
    # Block size configuration based on performance considerations
    BLOCK_SIZE_512 = 256
    BLOCK_SIZE_128 = 128
    
    # Calculate grid dimensions
    grid = lambda meta: (
        batch_size,
        seq_len,
    )
    
    # Launch optimized kernel
    optimized_silu_split_unsqueeze_kernel[grid](
        in_1_ptr=in_1,
        out_3_ptr=out_3,
        out_4_ptr=out_4,
        out_6_ptr=out_6,
        batch_size=batch_size,
        seq_len=seq_len,
        BLOCK_SIZE_512=BLOCK_SIZE_512,
        BLOCK_SIZE_128=BLOCK_SIZE_128
    )
    
    # Handle in_0 expansion
    out_7 = in_0[None, None, ...]
    
    return (out_7, out_3, out_6, out_4)

def replacement_func():
    return optimized_silu_split_unsqueeze