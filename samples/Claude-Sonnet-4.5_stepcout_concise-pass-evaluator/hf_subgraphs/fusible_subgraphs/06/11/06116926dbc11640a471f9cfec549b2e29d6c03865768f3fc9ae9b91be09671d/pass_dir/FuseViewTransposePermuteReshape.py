import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    """
    Pattern matching for view+transpose and permute+reshape operations.
    Must match exact structure in model.py - no cleanup statements like tmp_x = None
    This pattern matches face-parsing_start46_end50_12 with batch size 32
    """
    # First branch: in_1.view(32, -1, 1, 64).transpose(1, 2)
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    
    # Second branch: in_0.permute(0, 2, 1).reshape(32, 64, 128, 128)
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    
    return (tmp_1, tmp_3)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def fused_view_transpose_kernel(
    in_ptr,
    out_ptr,
    batch,
    seq_len,
    num_heads,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for view + transpose operation.
    Input: [batch, seq_len, num_heads * head_dim]
    Output: [batch, num_heads, seq_len, head_dim]
    """
    pid = tl.program_id(0)
    num_elements = batch * num_heads * seq_len * head_dim
    
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    
    # Calculate output indices
    b = offset // (num_heads * seq_len * head_dim)
    remainder = offset % (num_heads * seq_len * head_dim)
    h = remainder // (seq_len * head_dim)
    remainder = remainder % (seq_len * head_dim)
    s = remainder // head_dim
    d = remainder % head_dim
    
    # Calculate input indices
    # Input layout: [batch, seq_len, num_heads, head_dim]
    in_offset = b * (seq_len * num_heads * head_dim) + s * (num_heads * head_dim) + h * head_dim + d
    
    # Load and store
    data = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, data, mask=mask)


@triton.jit
def fused_permute_reshape_kernel(
    in_ptr,
    out_ptr,
    batch,
    seq_len,
    channels,
    h_out,
    w_out,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for permute + reshape operation.
    Input: [batch, seq_len, channels]
    After permute(0, 2, 1): [batch, channels, seq_len]
    Output: [batch, channels, h_out, w_out]
    """
    pid = tl.program_id(0)
    num_elements = batch * channels * h_out * w_out
    
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    
    # Output layout: [batch, channels, h_out, w_out]
    # After permute but before reshape: [batch, channels, seq_len]
    # Input layout: [batch, seq_len, channels]
    
    # Calculate output indices
    b = offset // (channels * h_out * w_out)
    remainder = offset % (channels * h_out * w_out)
    c = remainder // (h_out * w_out)
    spatial = remainder % (h_out * w_out)
    
    # Map to input: after permute it's [batch, channels, seq_len]
    # So for permute(0, 2, 1): output[b, c, s] = input[b, s, c]
    s = spatial  # seq_len position
    in_offset = b * (seq_len * channels) + s * channels + c
    
    # Load and store
    data = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, data, mask=mask)


@torch.fx.wrap
def fused_view_transpose_wrapper(in_tensor, num_heads, head_dim):
    """
    Wrapper for fused view + transpose operation.
    """
    batch, seq_len, _ = in_tensor.shape
    
    # Output shape: [batch, num_heads, seq_len, head_dim]
    out = torch.empty(batch, num_heads, seq_len, head_dim, 
                      dtype=in_tensor.dtype, device=in_tensor.device)
    
    BLOCK_SIZE = 1024
    num_elements = batch * num_heads * seq_len * head_dim
    grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_view_transpose_kernel[grid](
        in_tensor,
        out,
        batch,
        seq_len,
        num_heads,
        head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.fx.wrap
def fused_permute_reshape_wrapper(in_tensor, h_out, w_out):
    """
    Wrapper for fused permute + reshape operation.
    """
    batch, seq_len, channels = in_tensor.shape
    
    # Output shape: [batch, channels, h_out, w_out]
    out = torch.empty(batch, channels, h_out, w_out,
                      dtype=in_tensor.dtype, device=in_tensor.device)
    
    BLOCK_SIZE = 1024
    num_elements = batch * channels * h_out * w_out
    grid = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    fused_permute_reshape_kernel[grid](
        in_tensor,
        out,
        batch,
        seq_len,
        channels,
        h_out,
        w_out,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@torch.fx.wrap
def optimized_computation(in_0, in_1):
    """
    Optimized computation that fuses view+transpose and permute+reshape.
    """
    # Determine shapes dynamically
    batch = in_1.shape[0]
    seq_len = in_1.shape[1]
    total_dim = in_1.shape[2]
    
    # For in_1: view + transpose
    # Assuming the pattern is view(batch, -1, num_heads, head_dim).transpose(1, 2)
    # We need to infer num_heads and head_dim from the total dimension
    head_dim = 64 if total_dim % 64 == 0 else 32
    num_heads = total_dim // head_dim
    
    tmp_1 = fused_view_transpose_wrapper(in_1, num_heads, head_dim)
    
    # For in_0: permute + reshape
    channels = in_0.shape[2]
    seq_len_0 = in_0.shape[1]
    
    # Determine h_out and w_out from seq_len
    h_out = int(seq_len_0 ** 0.5)
    w_out = h_out
    
    tmp_3 = fused_permute_reshape_wrapper(in_0, h_out, w_out)
    
    return (tmp_1, tmp_3)


def replacement_func():
    return optimized_computation