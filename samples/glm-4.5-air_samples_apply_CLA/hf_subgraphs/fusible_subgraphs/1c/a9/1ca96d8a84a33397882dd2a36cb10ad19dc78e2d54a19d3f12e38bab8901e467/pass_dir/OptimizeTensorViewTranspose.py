import torch
import triton
import triton.language as tl

@triton.jit
def reshape_transpose_kernel(
    input_ptr,
    output_ptr,
    N, C_total, H, W,
    C_chunk: tl.constexpr, H_chunk: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    if pid >= N * C_total * H * W:
        return
        
    # Calculate indices
    batch_idx = pid // (C_total * H * W)
    c_out = (pid // (H * W)) % C_total
    h_out = (pid // W) % H
    w_out = pid % W
    
    # Map output back to input indices
    # Pattern: view(N, C_chunk, H_chunk, H, W) -> transpose(1,2) -> view(N, C_chunk*H_chunk, H, W)
    # output[c_out] corresponds to input[batch_idx, c_out // H_chunk, c_out % H_chunk, h_out, w_out]
    c_chunk = c_out // H_chunk
    h_chunk = c_out % H_chunk
    
    if batch_idx < N and c_chunk < C_chunk and h_chunk < H_chunk and h_out < H and w_out < W:
        input_idx = batch_idx * C_chunk * H_chunk * H * W + c_chunk * H_chunk * H * W + h_chunk * H * W + h_out * W + w_out
        output_idx = pid
        input_val = tl.load(input_ptr + input_idx)
        tl.store(output_ptr + output_idx, input_val)

@torch.fx.wrap
def optimized_reshape_transpose(input_tensor, N, C_chunk, H_chunk, H, W):
    # Original shape after concatenation: [N, C_total, H, W] where C_total = C_chunk * H_chunk
    C_total = C_chunk * H_chunk
    
    # Check if input is already in correct shape
    if input_tensor.shape == (N, C_total, H, W):
        # No transformation needed, just return a view
        return input_tensor
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate grid size
    BLOCK_SIZE = 256
    grid_size = N * C_total * H * W
    num_programs = (grid_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    reshape_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        N=N, C_total=C_total, H=H, W=W,
        C_chunk=C_chunk, H_chunk=H_chunk,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(x, N, C_chunk, H_chunk, H, W):
    # This pattern matches: view + transpose + contiguous + view
    # x has shape [N, C_total, H, W] where C_total = C_chunk * H_chunk
    
    # Original sequence:
    # tmp_7 = x.view(N, C_chunk, H_chunk, H, W)        # [N, C_chunk, H_chunk, H, W]
    # tmp_8 = torch.transpose(tmp_7, 1, 2)             # [N, H_chunk, C_chunk, H, W]
    # tmp_9 = tmp_8.contiguous()                       # make contiguous
    # tmp_10 = tmp_9.view(N, C_chunk * H_chunk, H, W)  # [N, C_total, H, W]
    
    # This sequence maps each output element (b, c, h, w) to input element (b, c // H_chunk, c % H_chunk, h, w)
    
    # Create the optimized transformation
    tmp_7_optimized = optimized_reshape_transpose(x, N, C_chunk, H_chunk, H, W)
    return tmp_7_optimized

def replacement_args(x, N, C_chunk, H_chunk, H, W):
    return (x, N, C_chunk, H_chunk, H, W)

def replacement_func():
    return optimized_reshape_transpose