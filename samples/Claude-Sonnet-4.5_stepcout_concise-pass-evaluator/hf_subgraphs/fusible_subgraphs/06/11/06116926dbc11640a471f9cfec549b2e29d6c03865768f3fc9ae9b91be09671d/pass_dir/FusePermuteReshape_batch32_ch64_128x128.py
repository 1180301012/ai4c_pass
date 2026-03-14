import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Pattern for permute + reshape on in_0
    Matches: in_0.permute(0, 2, 1).reshape(batch, channels, h, w)
    """
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    return tmp_3


def replacement_args(in_0):
    return (in_0,)


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
    Output: [batch, channels, h_out, w_out] where seq_len = h_out * w_out
    """
    pid = tl.program_id(0)
    num_elements = batch * channels * h_out * w_out
    
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < num_elements
    
    # Calculate output indices: [batch, channels, h_out, w_out]
    b = offset // (channels * h_out * w_out)
    remainder = offset % (channels * h_out * w_out)
    c = remainder // (h_out * w_out)
    spatial = remainder % (h_out * w_out)
    
    # Map to input: [batch, seq_len, channels]
    # After permute(0, 2, 1): out[b, c, s] = in[b, s, c]
    s = spatial  # spatial index maps to seq_len
    in_offset = b * (seq_len * channels) + s * channels + c
    
    # Load and store
    data = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offset, data, mask=mask)


@torch.fx.wrap
def fused_permute_reshape(in_tensor, batch, seq_len, channels, h_out, w_out):
    """
    Optimized fused permute + reshape operation.
    """
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


def replacement_func():
    def optimized(in_0):
        # Extract shape information
        batch = 32
        seq_len = in_0.shape[1]
        channels = 64
        h_out = 128
        w_out = 128
        return fused_permute_reshape(in_0, batch, seq_len, channels, h_out, w_out)
    
    return optimized