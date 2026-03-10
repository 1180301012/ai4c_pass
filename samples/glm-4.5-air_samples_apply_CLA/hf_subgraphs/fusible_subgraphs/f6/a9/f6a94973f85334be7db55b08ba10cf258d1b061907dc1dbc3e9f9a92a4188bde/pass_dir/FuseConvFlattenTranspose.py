import torch
import triton
import triton.language as tl

def pattern(x, weight):
    tmp_8 = torch.conv2d(x, weight, None, (16, 16), (0, 0), (1, 1), 1)
    tmp_9 = tmp_8.flatten(2)
    tmp_10 = tmp_9.transpose(1, 2)
    return tmp_10

def replacement_args(x, weight):
    return (x, weight)

@triton.jit
def conv_flatten_transpose_kernel(
    x_ptr, 
    weight_ptr, 
    out_ptr,
    N: tl.constexpr,
    C: tl.constexpr, 
    H_in: tl.constexpr,
    W_in: tl.constexpr,
    K: tl.constexpr,
    H_out: tl.constexpr,
    W_out: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one position in the flattened sequence
    pid = tl.program_id(0)
    
    # Calculate spatial position for this program
    h_idx = pid // W_out
    w_idx = pid % W_out
    
    # Process each output channel
    for c in range(N):
        # Initialize accumulator for this channel
        acc = 0.0
        
        # Convolution computation at this spatial position
        for kh in range(K):
            for kw in range(K):
                for ic in range(C):
                    # Input coordinates
                    ih = h_idx + kh
                    iw = w_idx + kw
                    
                    # Check bounds
                    if ih < H_in and iw < W_in:
                        # Load input value
                        x_val = tl.load(x_ptr + ic * H_in * W_in + ih * W_in + iw)
                        
                        # Load kernel weight
                        w_val = tl.load(weight_ptr + c * C * K * K + ic * K * K + kh * K + kw)
                        
                        # Multiply and accumulate
                        acc += x_val * w_val
        
        # Store result at the corresponding position in output sequence
        out_offset = c * H_out * W_out + pid
        tl.store(out_ptr + out_offset, acc)

@torch.fx.wrap
def fused_conv_flatten_transpose(x, weight):
    # Get input dimensions
    N, C, H_in, W_in = x.shape
    K = 16  # Kernel size from pattern
    H_out = H_in - K + 1  # 224 - 16 + 1 = 209
    W_out = W_in - K + 1  # 224 - 16 + 1 = 209
    total_elements = H_out * W_out
    
    # Create output tensor with shape [batch, seq_len, embed_dim]
    out = torch.empty((1, total_elements, N), dtype=x.dtype, device=x.device)
    
    # Reshape input and output for kernel (flatten batch dimension)
    x_flat = x.reshape(-1)  # [C * H_in * W_in]
    out_flat = out.reshape(-1)  # [N * H_out * W_out]
    
    # Launch kernel - each program handles one spatial position
    grid = (total_elements,)
    
    # Launch kernel
    conv_flatten_transpose_kernel[grid](
        x_ptr=x_flat,
        weight_ptr=weight,
        out_ptr=out_flat,
        N=N,
        C=C,
        H_in=H_in,
        W_in=W_in, 
        K=K,
        H_out=H_out,
        W_out=W_out,
        BLOCK_SIZE=1,  # Each program is independent
    )
    
    return out

def replacement_func():
    return fused_conv_flatten_transpose