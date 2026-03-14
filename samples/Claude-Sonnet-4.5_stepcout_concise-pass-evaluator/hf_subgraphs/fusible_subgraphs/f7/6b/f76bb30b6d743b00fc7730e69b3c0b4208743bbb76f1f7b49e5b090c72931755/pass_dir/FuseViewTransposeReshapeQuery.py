import torch
import triton
import triton.language as tl


def pattern(input_tensor):
    """
    Pattern: view -> transpose -> reshape
    This matches the transformation chain for query_states processing (seq_len=1)
    """
    tmp_7 = input_tensor.view(1, 1, 16, 64)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = tmp_8.reshape(16, -1, 64)
    return tmp_9


def replacement_args(input_tensor):
    return (input_tensor,)


@triton.jit
def fused_view_transpose_reshape_query_kernel(
    input_ptr,
    output_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that performs: view(1,1,16,64) -> transpose(1,2) -> reshape(16,-1,64)
    
    Input shape: [1, 1, 1024]
    After view: [1, 1, 16, 64]
    After transpose(1,2): [1, 16, 1, 64]
    After reshape: [16, 1, 64]
    
    Direct mapping:
    output[head, 0, feat] = input[0, 0, head * 64 + feat]
    """
    # Get program ID (one per head)
    head_id = tl.program_id(0)
    
    # Load a block of features for this head
    feat_offsets = tl.arange(0, BLOCK_SIZE)
    mask = feat_offsets < 64
    
    # Input layout: [1, 1, 1024]
    # For head h, we need features from input[0, 0, h*64:(h+1)*64]
    input_base = head_id * 64
    input_offsets = input_base + feat_offsets
    
    # Load input data
    data = tl.load(input_ptr + input_offsets, mask=mask, other=0.0)
    
    # Output layout: [16, 1, 64]
    # For head h, feat f: output[h, 0, f]
    output_base = head_id * 64
    output_offsets = output_base + feat_offsets
    
    # Store output data
    tl.store(output_ptr + output_offsets, data, mask=mask)


@torch.fx.wrap
def fused_view_transpose_reshape_query(input_tensor):
    """
    Wrapper function for the fused kernel
    Input: [1, 1, 1024]
    Output: [16, 1, 64]
    """
    batch, seq_len, hidden = input_tensor.shape
    assert batch == 1 and seq_len == 1 and hidden == 1024, f"Expected [1, 1, 1024], got {input_tensor.shape}"
    
    # Output shape: [16, 1, 64]
    output = torch.empty((16, 1, 64), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel with 1D grid: (16 heads)
    BLOCK_SIZE = 64
    grid = (16,)
    
    fused_view_transpose_reshape_query_kernel[grid](
        input_tensor,
        output,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


def replacement_func():
    return fused_view_transpose_reshape_query