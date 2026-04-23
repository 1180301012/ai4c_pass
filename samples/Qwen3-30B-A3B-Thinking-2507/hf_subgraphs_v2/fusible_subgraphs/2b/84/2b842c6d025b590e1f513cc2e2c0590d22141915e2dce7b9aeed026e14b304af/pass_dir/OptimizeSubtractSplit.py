import torch
import triton
import triton.language as tl

def pattern(in0, in1):
    tmp1 = in0 * 1000000.0
    tmp2 = in1 - tmp1
    split = tmp2.split(1, dim=-1)
    tmp4 = split[0]
    tmp5 = split[1]
    tmp6 = tmp4.squeeze(-1)
    tmp7 = tmp6.contiguous()
    tmp8 = tmp5.squeeze(-1)
    tmp9 = tmp8.contiguous()
    return (tmp7, tmp9)

def replacement_args(in0, in1):
    return (in0, in1)

@triton.jit
def optimized_kernel(
    in0_ptr,
    in1_ptr,
    out1_ptr,
    out2_ptr,
    seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    seq_id = tl.program_id(0)
    start = seq_id * BLOCK_SIZE
    end = min(start + BLOCK_SIZE, seq_len)
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < seq_len
    
    in0_val = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    scaled = in0_val * 1000000.0
    
    in1_val0 = tl.load(in1_ptr + offsets * 2, mask=mask, other=0.0)
    in1_val1 = tl.load(in1_ptr + offsets * 2 + 1, mask=mask, other=0.0)
    
    out0 = in1_val0 - scaled
    out1 = in1_val1 - scaled
    
    tl.store(out1_ptr + offsets, out0, mask=mask)
    tl.store(out2_ptr + offsets, out1, mask=mask)

@torch.fx.wrap
def kernel_wrapper(in0, in1):
    batch, seq_len, _ = in1.shape
    assert batch == 1, "Batch size must be 1"
    
    out1 = torch.empty([1, seq_len], dtype=in1.dtype, device=in1.device)
    out2 = torch.empty([1, seq_len], dtype=in1.dtype, device=in1.device)
    
    BLOCK_SIZE = 32
    num_blocks = (seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_kernel[(num_blocks,)](
        in0_ptr=in0.data_ptr(),
        in1_ptr=in1.data_ptr(),
        out1_ptr=out1.data_ptr(),
        out2_ptr=out2.data_ptr(),
        seq_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return (out1, out2)

def replacement_func():
    return kernel_wrapper