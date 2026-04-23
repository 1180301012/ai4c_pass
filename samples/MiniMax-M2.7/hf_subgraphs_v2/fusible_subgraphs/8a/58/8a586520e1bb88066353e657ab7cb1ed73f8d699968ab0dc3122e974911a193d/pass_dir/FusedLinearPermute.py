import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2, in_3):
    """
    Match the pattern: linear + permute(0, 3, 1, 2)
    
    Input shapes:
    - in_0 (bias): [16]
    - in_1 (weight): [16, 3]
    - in_2: [1, 16, 196, 48] - for transpose
    - in_3 (input): [1, 196, 196, 3]
    
    Output shapes:
    - tmp_3: [1, 16, 196, 196] (linear + permute)
    - tmp_4: [1, 16, 48, 196] (transpose)
    """
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.permute(0, 3, 1, 2)
    tmp_4 = in_2.transpose(-2, -1)
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3):
    """Extract arguments needed for the fused linear+permute kernel"""
    return (in_0, in_1, in_2, in_3)


# Detect if input is float32, float16, or bfloat16
DTYPE_FP32 = 0
DTYPE_FP16 = 1
DTYPE_BF16 = 2


@triton.jit
def fused_linear_permute_kernel(
    inp_ptr, w_ptr, bias_ptr, out_ptr,
    N, K, C,  # N=196*196, K=16, C=3
    dtype: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Fused kernel for linear(in_3, in_1, in_0) followed by permute(0, 3, 1, 2)
    
    Original: [1, 196, 196, 3] @ [16, 3]^T + [16] = [1, 196, 196, 16] -> permute -> [1, 16, 196, 196]
    
    We compute directly into the output layout [1, 16, 196, 196]
    Output index (b=0, k, i, j) corresponds to:
    - Original linear index (b=0, i, j, k)
    - inp[b, i, j, :] has shape [3]
    - weight[k, :] has shape [3]
    """
    # Program id: map to (k, i, j) in output space [16, 196, 196]
    pid = tl.program_id(0)
    
    # Calculate k, i, j from pid
    # Output shape: [1, 16, 196, 196], K=16, I=196, J=196
    k = pid // (196 * 196)
    remainder = pid % (196 * 196)
    i = remainder // 196
    j = remainder % 196
    
    # Determine accumulator dtype based on input dtype
    acc_dtype = tl.float32 if dtype == DTYPE_FP32 else tl.float32
    
    # Initialize accumulator for linear: sum over C
    acc = tl.zeros([1, 1], dtype=acc_dtype)
    
    # Pointer arithmetic:
    # inp[b, i, j, c] -> b * (196 * 196 * 3) + i * (196 * 3) + j * 3 + c
    # weight[k, c] -> k * 3 + c
    
    # Process in blocks of BLOCK_C
    for c_start in range(0, C, BLOCK_C):
        c_offsets = c_start + tl.arange(0, BLOCK_C)
        c_mask = c_offsets < C
        
        # Load input tile
        inp_offsets = i * 196 * 3 + j * 3 + c_offsets
        inp = tl.load(inp_ptr + inp_offsets, mask=c_mask, other=0.0).to(acc_dtype)
        
        # Load weight tile (weight is [K, C] layout)
        w_offsets = k * 3 + c_offsets
        w = tl.load(w_ptr + w_offsets, mask=c_mask, other=0.0).to(acc_dtype)
        
        # Accumulate matmul
        acc += tl.sum(inp * w, axis=0)
    
    # Add bias
    bias_val = tl.load(bias_ptr + k).to(acc_dtype)
    acc = acc + bias_val
    
    # Store result in [1, 16, 196, 196] layout
    # out[b, k, i, j] = acc
    out_offset = k * 196 * 196 + i * 196 + j
    if dtype == DTYPE_FP32:
        tl.store(out_ptr + out_offset, acc.to(tl.float32))
    elif dtype == DTYPE_FP16:
        tl.store(out_ptr + out_offset, acc.to(tl.float16))
    else:  # bfloat16
        tl.store(out_ptr + out_offset, acc.to(tl.bfloat16))


@triton.jit
def optimized_transpose_kernel(
    inp_ptr, out_ptr,
    B, M, N,  # [B, M, N] -> [B, N, M]
    dtype: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for transposing last two dimensions.
    Input: [B, M, N], Output: [B, N, M]
    """
    pid = tl.program_id(0)
    
    # Each program handles one element: out[b, n, m]
    b = pid // (N * M)
    remainder = pid % (N * M)
    n = remainder // M
    m = remainder % M
    
    # Load from inp[b, m, n]
    inp_offset = b * M * N + m * N + n
    val = tl.load(inp_ptr + inp_offset)
    
    # Store to out[b, n, m]
    out_offset = b * N * M + n * M + m
    tl.store(out_ptr + out_offset, val)


def get_dtype_enum(dtype):
    """Get Triton dtype enum for kernel"""
    if dtype == torch.float32:
        return DTYPE_FP32
    elif dtype == torch.float16:
        return DTYPE_FP16
    elif dtype == torch.bfloat16:
        return DTYPE_BF16
    else:
        return DTYPE_FP32


@torch.fx.wrap
def fused_kernel_wrapper(in_0, in_1, in_2, in_3):
    """
    Wrapper function that launches fused kernels for:
    1. Linear + Permute: [1, 196, 196, 3] -> [1, 16, 196, 196]
    2. Transpose: [1, 16, 196, 48] -> [1, 16, 48, 196]
    """
    # Move weight and bias to GPU if needed, and ensure in_3 is also on GPU
    weight = in_1
    bias = in_0
    inp = in_3
    
    # Handle CPU tensors (weight and bias are on CPU per weight_meta)
    if weight.device.type == 'cpu':
        weight = weight.cuda(non_blocking=True)
    if bias.device.type == 'cpu':
        bias = bias.cuda(non_blocking=True)
    if inp.device.type == 'cpu':
        inp = inp.cuda(non_blocking=True)
    
    # Synchronize to ensure GPU tensors are ready
    torch.cuda.synchronize()
    
    # Determine dtype for kernel
    dtype_enum = get_dtype_enum(inp.dtype)
    
    # ========== Fused Linear + Permute ==========
    # Input: [1, 196, 196, 3] = NHWC format
    # Weight: [16, 3]
    # Output: [1, 16, 196, 196]
    
    num_programs = 16 * 196 * 196  # 614656 programs
    
    # Output tensor - match input dtype and device
    out_linear = torch.empty([1, 16, 196, 196], dtype=inp.dtype, device=inp.device)
    
    # Launch kernel
    fused_linear_permute_kernel[(num_programs,)](
        inp_ptr=inp,
        w_ptr=weight,
        bias_ptr=bias,
        out_ptr=out_linear,
        N=196 * 196,
        K=16,
        C=3,
        dtype=dtype_enum,
        BLOCK_C=16,  # Load all 3 elements at once
    )
    
    # ========== Optimized Transpose ==========
    # Input: [1, 16, 196, 48] (contiguous in memory as NCHW)
    # Output: [1, 16, 48, 196]
    
    # Ensure in_2 is on GPU
    inp2 = in_2
    if inp2.device.type == 'cpu':
        inp2 = inp2.cuda(non_blocking=True)
    
    torch.cuda.synchronize()
    
    out_transpose = torch.empty([1, 16, 48, 196], dtype=inp2.dtype, device=inp2.device)
    
    num_programs_transpose = 1 * 16 * 48 * 196
    optimized_transpose_kernel[(num_programs_transpose,)](
        inp_ptr=inp2,
        out_ptr=out_transpose,
        B=1, M=48, N=196,
        dtype=dtype_enum,
        BLOCK_SIZE=1,
    )
    
    torch.cuda.synchronize()
    
    return (out_linear, out_transpose)


def replacement_func():
    return fused_kernel_wrapper