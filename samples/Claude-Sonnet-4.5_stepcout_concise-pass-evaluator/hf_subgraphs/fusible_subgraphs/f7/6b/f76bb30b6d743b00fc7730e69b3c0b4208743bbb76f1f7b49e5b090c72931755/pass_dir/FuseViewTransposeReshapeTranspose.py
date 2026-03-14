import torch
import triton
import triton.language as tl


def pattern(input_tensor):
    """
    Pattern: view -> transpose -> reshape -> transpose
    This matches the transformation chain for key_states processing
    """
    tmp_3 = input_tensor.view(1, -1, 16, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_10 = tmp_4.reshape(16, -1, 64)
    tmp_12 = tmp_10.transpose(1, 2)
    return tmp_12


def replacement_args(input_tensor):
    return (input_tensor,)


@triton.jit
def fused_view_transpose_reshape_transpose_kernel(
    input_ptr,
    output_ptr,
    seq_len: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs: view(1,-1,16,64) -> transpose(1,2) -> reshape(16,-1,64) -> transpose(1,2)
    
    Input shape: [1, seq_len, 1024]
    After view: [1, seq_len, 16, 64]
    After first transpose(1,2): [1, 16, seq_len, 64]
    After reshape: [16, seq_len, 64]
    After second transpose(1,2): [16, 64, seq_len]
    
    We can compute the direct mapping:
    output[head, feat, seq] = input[0, seq, head * 64 + feat]
    """
    # Get program ID
    head_id = tl.program_id(0)
    seq_id = tl.program_id(1)
    
    # Each program handles one position
    if seq_id < seq_len:
        # Load a block of features for this head and sequence position
        feat_offsets = tl.arange(0, BLOCK_SIZE)
        mask = feat_offsets < 64
        
        # Input layout: [1, seq_len, 1024] -> flatten to [seq_len, 1024]
        # For head h and seq s, we need features from input[s, h*64:(h+1)*64]
        input_base = seq_id * 1024 + head_id * 64
        input_offsets = input_base + feat_offsets
        
        # Load input data
        data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
        
        # Output layout: [16, 64, seq_len]
        # For head h, feat f, seq s: output[h, f, s]
        output_base = head_id * 64 * seq_len + seq_id
        output_offsets = output_base + feat_offsets * seq_len
        
        # Store output data
        tl.store(output_ptr + output_offsets, data, mask=mask)


@torch.fx.wrap
def fused_view_transpose_reshape_transpose(input_tensor):
    """
    Wrapper function for the fused kernel
    Input: [1, seq_len, 1024]
    Output: [16, 64, seq_len]
    """
    batch, seq_len, hidden = input_tensor.shape
    assert batch == 1 and hidden == 1024, f"Expected [1, seq_len, 1024], got {input_tensor.shape}"
    
    # Output shape: [16, 64, seq_len]
    output = torch.empty((16, 64, seq_len), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel with 2D grid: (16 heads, seq_len positions)
    BLOCK_SIZE = 64
    grid = (16, seq_len)
    
    fused_view_transpose_reshape_transpose_kernel[grid](
        input_tensor,
        output,
        seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_view_transpose_reshape_transpose