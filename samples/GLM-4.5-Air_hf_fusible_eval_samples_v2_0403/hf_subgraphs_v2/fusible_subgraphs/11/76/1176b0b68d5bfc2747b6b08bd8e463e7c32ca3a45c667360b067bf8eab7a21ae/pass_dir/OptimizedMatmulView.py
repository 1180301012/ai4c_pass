import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(a, b):
    """Match matmul followed by view operation"""
    result = a @ b
    # Infer the view shape from the original pattern
    # The view shape will be determined by the specific tensor dimensions
    return result

def replacement_args(a, b):
    """Extract arguments for the replacement kernel"""
    return (a, b)

# Optimized Triton kernel for matmul + view fusion
@triton.jit
def optimized_matmul_view_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N, K,  # Matrix dimensions
    view_shape1, view_shape2, view_shape3, view_shape4,  # Target view shape
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Optimized kernel for matrix multiplication with direct output in view shape"""
    
    # Program ID
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    m = pid % grid_m
    n = (pid // grid_m) % grid_n
    
    # Pointer offsets for the block
    a_ptrs = a_ptr + m * BLOCK_SIZE_M * K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :] + n * BLOCK_SIZE_N * K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks with masking
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K - k), other=0.0)
        
        # Matrix multiply-add
        accumulator += tl.dot(a, b, out_dtype=tl.float32)
        
        # Update pointers for next iteration
        a_ptrs += BLOCK_SIZE_K * K
        b_ptrs += BLOCK_SIZE_K * BLOCK_SIZE_N
    
    # Store result with output in desired view shape
    output_block = accumulator.to(tl.float16)
    output_offset = m * BLOCK_SIZE_M * N + n * BLOCK_SIZE_N
    
    # Store the block
    tl.store(output_ptr + output_offset, output_block.to(tl.float16))

@triton.jit
def optimized_matmul_view_kernel_bf16(
    a_ptr, b_ptr, output_ptr,
    M, N, K,
    view_shape1, view_shape2, view_shape3, view_shape4,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """BF16 optimized version"""
    pid = tl.program_id(0)
    grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    m = pid % grid_m
    n = (pid // grid_m) % grid_n
    
    a_ptrs = a_ptr + m * BLOCK_SIZE_M * K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    b_ptrs = b_ptr + tl.arange(0, BLOCK_SIZE_N)[None, :] + n * BLOCK_SIZE_N * K + tl.arange(0, BLOCK_SIZE_K)[:, None]
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.bfloat16)
    
    for k in range(0, K, BLOCK_SIZE_K):
        a = tl.load(a_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(tl.arange(0, BLOCK_SIZE_K)[:, None] < K - k), other=0.0)
        
        accumulator += tl.dot(a, b, out_dtype=tl.bfloat16)
        
        a_ptrs += BLOCK_SIZE_K * K
        b_ptrs += BLOCK_SIZE_K * BLOCK_SIZE_N
    
    output_block = accumulator
    output_offset = m * BLOCK_SIZE_M * N + n * BLOCK_SIZE_N
    
    tl.store(output_ptr + output_offset, output_block)

# Optimized kernel wrapper
@torch.fx.wrap
def optimized_matmul_view(a, b, target_view_shape):
    """High-performance matmul + view fusion"""
    if target_view_shape is None:
        target_view_shape = a.shape[:-1] + (a.shape[-1],)  # Default to keeping dims
    
    # Determine matrix dimensions from input shapes
    if len(a.shape) == 4 and len(b.shape) == 4:
        # (B,C,O,I) @ (B,C,I,H,W) pattern - need to reshape for proper matrix mult
        B, C, O, I = a.shape
        _, _, _, H_in, W_in = b.shape
        
        # Reshape for matrix multiplication: (B*C, O, I) @ (B*C, I, H*W)
        a_flat = a.reshape(-1, O, I)
        b_flat = b.reshape(-1, I, H_in * W_in)
        
        M = B * C * O
        N = H_in * W_in
        K = I
        
        output_shape = (B * C, O, N)
        
    else:
        # Standard matmul case: (B,H,O,I) @ (B,H,I,1) -> (B,H,O,1)
        # Handle different tensor patterns
        if len(b.shape) == 4 and b.shape[-1] == 1:
            # Attention-like pattern
            batch_size = a.shape[0]
            heads = a.shape[1] if len(a.shape) > 2 else 1
            output_features = a.shape[-2]
            input_seq = a.shape[-1]
            
            B = batch_size * heads
            M = B * output_features
            N = 1
            K = input_seq
            
            # Reshape inputs
            a_flat = a.reshape(B, output_features, input_seq)
            b_flat = b.reshape(B, input_seq, N)
            
            output_shape = (B, output_features, N)
            
        else:
            # Fallback to standard matmul
            a_flat = a
            b_flat = b
            if a_flat.dim() > 2:
                M = a_flat.numel() // (a_flat.shape[-1])
                K = a_flat.shape[-1]
                N = b_flat.shape[-1]
                output_shape = a_flat.shape[:-1] + (N,)
            else:
                M, K = a_flat.shape
                N = b_flat.shape[-1]
                output_shape = (M, N)
    
    # Check data type and select appropriate kernel
    if a.dtype == torch.bfloat16:
        kernel = optimized_matmul_view_kernel_bf16
        dtype = torch.bfloat16
    else:
        kernel = optimized_matmul_view_kernel
        dtype = torch.float16
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=dtype, device=a.device)
    
    # Configure block sizes based on matrix dimensions
    if N > 512:
        block_size_n = 128
    else:
        block_size_n = min(N, 64)
    
    if M > 512:
        block_size_m = 64
    else:
        block_size_m = min(M, 32)
    
    block_size_k = min(32, K)
    
    # Calculate grid size
    grid_m = (M + block_size_m - 1) // block_size_m
    grid_n = (N + block_size_n - 1) // block_size_n
    grid_size = grid_m * grid_n
    
    # Launch kernel
    kernel[grid_size](
        a_flat, b_flat, output,
        M, N, K,
        *target_view_shape if len(target_view_shape) == 4 else target_view_shape + (1, 1),
        block_size_m, block_size_n, block_size_k
    )
    
    # Apply final view operation if needed
    if target_view_shape is not None and len(target_view_shape) == 4:
        if len(output_shape) == 3:
            # (B*C, O, H*W) -> (B, C, O, H, W) -> (B, C*O, H, W)
            total_elements = output.numel()
            B, C, O, H_in, W_in = target_view_shape
            if B * C * O * H_in * W_in == total_elements:
                output = output.reshape(B, C, O, H_in, W_in)
                output = output.reshape(B, C * O, H_in, W_in)
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    def optimized_func(a, b):
        # Infer target view from typical patterns
        if len(a.shape) == 4 and len(b.shape) == 4:
            # YOLO-like pattern
            B, C, O, I = a.shape
            _, _, _, H_in, W_in = b.shape
            target_view = (B, C * O, H_in, W_in)
        elif len(b.shape) == 4 and b.shape[-1] == 1:
            # Attention-like pattern
            B, H, O, I = a.shape
            target_view = (B, H * O, 1, 1)
        else:
            target_view = None
        
        return optimized_matmul_view(a, b, target_view)
    
    return optimized_func