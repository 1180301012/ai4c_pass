import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp_3 = in_3.mean(-2)
    return tmp_3

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def mean_kernel(
    x_ptr,           # input tensor (B, 49, 448)
    out_ptr,         # output tensor (B, 448)
    B: tl.constexpr, # batch size
    D: tl.constexpr, # feature dimension (448)
    SEQUENCE_LEN: tl.constexpr,  # sequence length (49)
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_D: tl.constexpr,
):
    # Program ID determines which batch and feature this program computes
    row = tl.program_id(0)  # batch index
    col = tl.program_id(1)  # feature index
    
    # Initialize accumulator
    acc = 0.0
    
    # Compute mean by iterating over sequence dimension
    for seq in tl.range(0, SEQUENCE_LEN):
        # Load element: (batch, seq, feature)
        src_ptr_row = x_ptr + row * (49 * 448)  # batch stride
        src_ptr_col = src_ptr_row + seq * 448    # sequence stride
        src_ptr = src_ptr_col + col              # feature index
        
        x = tl.load(src_ptr, mask=True, other=0.0)
        acc += x
    
    # Divide by sequence length to get mean
    mean_val = acc / SEQUENCE_LEN
    
    # Store result: (batch, feature)
    out_ptr_row = out_ptr + row * 448
    tl.store(out_ptr_row + col, mean_val)

@torch.fx.wrap
def triton_mean(in_3):
    B = in_3.shape[0]  # batch size (1, 64, or 256)
    SEQUENCE_LEN = 49  # fixed sequence length dimension
    D = 448            # feature dimension
    
    out = torch.empty((B, D), dtype=torch.float32, device=in_3.device)
    
    BLOCK_SIZE_B = 64    # Process 64 batch items per program
    BLOCK_SIZE_D = 448   # Process all features per program for simplicity
    
    num_B = (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    num_D = (D + BLOCK_SIZE_D - 1) // BLOCK_SIZE_D
    
    mean_kernel[(num_B, num_D)](
        in_3,
        out,
        B,
        D,
        SEQUENCE_LEN,
        BLOCK_SIZE_B,
        BLOCK_SIZE_D,
    )
    
    return out

def replacement_func():
    return triton_mean