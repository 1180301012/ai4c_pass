import torch
import triton
import triton.language as tl


def pattern(input_tensor):
    """
    Pattern: view -> transpose -> reshape
    This matches the transformation chain for value_states processing from linear output
    """
    tmp_5 = input_tensor.view(1, -1, 16, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    tmp_11 = tmp_6.reshape(16, -1, 64)
    return tmp_11


def replacement_args(input_tensor):
    return (input_tensor,)


@triton.jit
def fused_view_transpose_reshape_kernel(
    input_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs: view(1,-1,16,64) -> transpose(1,2) -> reshape(16,-1,64)
    
    Input shape: [1, seq_len, 1024]
    After view: [1, seq_len, 16, 64]
    After transpose(1,2): [1, 16, seq_len, 64]
    After reshape: [16, seq_len, 64]
    
    Direct mapping:
    output[head, seq, feat] = input[0, seq, head * 64 + feat]
    """
    # Get program ID
    head_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Each program handles one position
    if seq_id < seq_len:
        # Load a block of features for this head and sequence position
        feat_offsets = tl.arange(0, BLOCK_SIZE)
        mask = feat_offsets < 64
        
        # Input layout: [1, seq_len, 1024]
        # For head h and seq s, we need features from input[0, s, h*64:(h+1)*64]
        input_base = seq_id * 1024 + head_id * 64
        input_offsets = input_base + feat_offsets
        
        # Load input data
        data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Output layout: [16, seq_len, 64]
        # For head h, seq s, feat f: output[h, s, f]
        output_base = head_id * seq_len * 64 + seq_id * 64
        output_offsets = output_base + feat_offsets
        
        # Store output data
        tl.store(output_ptr + output_offsets, data, mask=mask)


@torch.fx.wrap
def fused_view_transpose_reshape(input_tensor):
    """
    Wrapper function for the fused kernel
    Input: [1, seq_len, 1024]
    Output: [16, seq_len, 64]
    """
    batch, seq_len, hidden = input_tensor.shape
    assert batch == 1 and hidden == 1024, f"Expected [1, seq_len, 1024], got {input_tensor.shape}"
    
    # Output shape: [16, seq_len, 64]
    output = torch.empty((16, seq_len, 64), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel with 2D grid: (16 heads, seq_len positions)
    BLOCK_SIZE = 64
    grid = (16, seq_len)
    
    fused_view_transpose_reshape_kernel[grid](
        input_tensor,
        output,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_view_transpose_reshape