import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = torch.nn.functional.linear(in_2, tmp_1, tmp_0)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@triton.jit
def linear_kernel(
    x_ptr,           # input matrix (B, 448)
    w_ptr,           # weight matrix (2, 448)  
    b_ptr,           # bias vector (2,)
    out_ptr,         # output matrix (B, 2)
    B: tl.constexpr, # batch size
    BLOCK_SIZE_B: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID determines which rows of the output this program computes
    row = tl.program_id(0)
    col = tl.program_id(1)
    
    # Initialize output
    acc = tl.zeros((), dtype=tl.float32)
    
    # Compute matrix multiplication
    for k in tl.range(0, 448, BLOCK_SIZE_K):
        # Load input row (current batch item)
        x_ptr_row = x_ptr + row * 448
        x = tl.load(x_ptr_row + k, mask=k < 448, other=0.0)
        
        # Load weight column (current output feature)
        w_ptr_col = w_ptr + col * 448
        w = tl.load(w_ptr_col + k, mask=k < 448, other=0.0)
        
        # Multiply and accumulate
        acc += x * w
    
    # Add bias
    b = tl.load(b_ptr + col)
    acc += b
    
    # Store result
    out_ptr_row = out_ptr + row * 2
    tl.store(out_ptr_row + col, acc)

@torch.fx.wrap
def triton_linear(in_2, in_1, in_0):
    B = in_2.shape[0]  # batch size
    out = torch.empty((B, 2), dtype=torch.float32, device=in_2.device)
    
    BLOCK_SIZE_B = 64   # Process 64 rows per program for good GPU occupancy
    BLOCK_SIZE_K = 256  # Process 256 elements per loop for good memory coalescing
    
    num_B = (B + BLOCK_SIZE_B - 1) // BLOCK_SIZE_B
    num_K = 2  # Each program computes output for one of the 2 output features
    
    linear_kernel[(num_B, num_K)](
        in_2,
        in_1.t(),  # Transpose weights to (448, 2) for column-major access
        in_0,
        out,
        B,
        BLOCK_SIZE_B,
        BLOCK_SIZE_K,
    )
    
    return out

def replacement_func():
    return triton_linear