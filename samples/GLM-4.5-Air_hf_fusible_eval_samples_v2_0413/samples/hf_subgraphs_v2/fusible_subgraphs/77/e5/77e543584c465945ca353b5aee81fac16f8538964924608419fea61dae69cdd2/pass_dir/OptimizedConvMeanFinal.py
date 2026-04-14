import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    # Simple pattern that matches conv2d + mean structure
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 1)
    tmp_2 = conv2d.mean((2, 3), keepdim=True)
    return (conv2d, tmp_2)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def optimized_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    z_ptr,
    M, N, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Simple matrix multiplication with mean computation
    start_n = pid_n * BLOCK_SIZE
    end_n = tl.minimum((pid_n + 1) * BLOCK_SIZE, N)
    
    acc = 0.0
    for k in range(H * W):
        x_offset = pid_m * N * H * W + start_n * H * W + k
        w_offset = start_n * H * W + k
        y_offset = pid_m * end_n * H * W + (start_n - pid_n * BLOCK_SIZE) * H * W + k
        
        x_val = tl.load(x_ptr + x_offset).to(tl.float32)
        w_val = tl.load(w_ptr + w_offset).to(tl.float32)
        y_val = x_val * w_val
        
        tl.store(y_ptr + y_offset, y_val.to(x_ptr.type.element_ty))
        acc += y_val
    
    # Mean computation  
    if pid_m == 0 and pid_n == 0:
        mean_val = acc / (H * W * N)
        tl.store(z_ptr, mean_val.to(x_ptr.type.element_ty))

@torch.fx.wrap
def optimized_function(in_0, in_1):
    # Extract basic tensor information
    N, C_in, H, W = in_1.shape
    C_out = in_0.shape[0]
    
    # Create outputs
    conv_out = torch.empty(N, C_out, H, W, dtype=in_1.dtype, device=in_1.device)
    mean_out = torch.empty(N, C_out, 1, 1, dtype=in_1.dtype, device=in_1.device)
    
    # Launch optimized kernel
    BLOCK_SIZE = 64
    grid = (N, (C_out + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    optimized_kernel[grid](
        in_1, in_0, conv_out, mean_out,
        N, C_out, H, W,
        BLOCK_SIZE
    )
    
    # Final mean computation
    for n in range(N):
        for c in range(C_out):
            mean_val = conv_out[n, c].mean()
            mean_out[n, c, 0, 0] = mean_val
    
    return conv_out, mean_out

def replacement_func():
    return optimized_function