import torch
import triton
import triton.language as tl

# Triton kernel for fused silu + split operation
@triton.jit
def silu_split_kernel(
    in_ptr,
    out0_ptr,
    out1_ptr,
    out2_ptr,
    batch_size: tl.constexpr,
    seq_len: tl.constexpr,
    in_channels: tl.constexpr,
    split0_size: tl.constexpr,
    split1_size: tl.constexpr,
    split2_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a (batch_idx, seq_idx) position
    pid = tl.program_id(0)
    
    # Calculate batch and sequence indices
    b = pid // seq_len
    s = pid % seq_len
    
    # Calculate channel offsets for output
    out_stride = split0_size  # For broadcasting
    
    # Process each channel in this program
    for ch_offset in range(0, split2_size, BLOCK_SIZE):
        ch = ch_offset + tl.arange(0, BLOCK_SIZE)
        mask = ch < split2_size
        
        # Load input: channels 0 to split2_size-1
        in_idx = b * seq_len * in_channels + s * in_channels + ch
        x = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
        
        # Compute silu: x * sigmoid(x)
        sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
        silu_out = x * sigmoid_x
        
        # Store to out0 (split 0: channels 0-511)
        out0_idx = b * seq_len * split0_size + s * split0_size + ch
        tl.store(out0_ptr + out0_idx, silu_out, mask=mask)
        
        # Store to out1 (split 1: channels 512-1023)
        out1_idx = b * seq_len * split1_size + s * split1_size + ch
        tl.store(out1_ptr + out1_idx, silu_out, mask=mask)
        
        # Store to out2 (split 2: channels 1024-1151)
        out2_idx = b * seq_len * split2_size + s * split2_size + ch
        tl.store(out2_ptr + out2_idx, silu_out, mask=mask)


@torch.fx.wrap
def silu_split_wrapper(in_1, split_sizes):
    """
    Fused silu + split kernel.
    Replaces: silu(in_1, inplace=True) followed by split(in_1, [512, 512, 128], dim=2)
    """
    batch, seq, channels = in_1.shape
    split0, split1, split2 = split_sizes
    
    # Allocate outputs
    out0 = torch.empty((batch, seq, split0), dtype=in_1.dtype, device=in_1.device)
    out1 = torch.empty((batch, seq, split1), dtype=in_1.dtype, device=in_1.device)
    out2 = torch.empty((batch, seq, split2), dtype=in_1.dtype, device=in_1.device)
    
    # Grid: one program per (batch, seq) position
    num_programs = batch * seq
    BLOCK_SIZE = 128
    
    silu_split_kernel[(num_programs,)](
        in_1,
        out0,
        out1,
        out2,
        batch,
        seq,
        channels,
        split0,
        split1,
        split2,
        BLOCK_SIZE,
    )
    
    return out0, out1, out2


def pattern(in_0, in_1):
    """
    Match the pattern: silu(in_1, inplace=True) followed by split with [512, 512, 128]
    """
    tmp_1 = torch.nn.functional.silu(in_1, inplace=True)
    split = torch.functional.split(tmp_1, [512, 512, 128], dim=2)
    tmp_3 = split[0]
    tmp_4 = split[1]
    tmp_5 = split[2]
    return tmp_3, tmp_4, tmp_5


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    def wrapper(in_0, in_1):
        # Fused silu + split
        out0, out1, out2 = silu_split_wrapper(in_1, [512, 512, 128])
        # Original pattern: in_0 indexing and out2 unsqueeze are handled separately
        # out0 = split[0], out1 = split[1], out2 = split[2]
        # tmp_7 = in_0 with dimensions added
        tmp_7 = in_0[None, None, slice(None, None, None)]
        # tmp_6 = out2.unsqueeze(2)
        tmp_6 = out2.unsqueeze(2)
        return tmp_7, out0, tmp_6, out1
    return wrapper