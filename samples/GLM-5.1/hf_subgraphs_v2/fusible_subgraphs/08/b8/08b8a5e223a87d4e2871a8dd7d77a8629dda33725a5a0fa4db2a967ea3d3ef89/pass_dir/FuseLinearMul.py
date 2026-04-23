import torch
import triton
import triton.language as tl

# Pattern for transformers/SwiGLU: linear + element-wise multiply fusion
# linear = F.linear(input, weight, None)
# result = other * linear
# return (result,)
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    return (tmp_2,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_fuse_linear_mul")

# Fused Linear + Multiply Triton kernel
@triton.jit
def fused_linear_mul_kernel(
    input_ptr, weight_ptr, other_ptr, output_ptr,
    M, N, K,
    stride_input_m, stride_input_k,
    stride_weight_n, stride_weight_k,
    stride_other_m, stride_other_n,
    stride_output_m, stride_output_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k_cur = k_start + offs_k

        input_ptrs = input_ptr + offs_m[:, None] * stride_input_m + offs_k_cur[None, :] * stride_input_k
        weight_ptrs = weight_ptr + offs_n[None, :] * stride_weight_n + offs_k_cur[:, None] * stride_weight_k

        input_mask = (offs_m[:, None] < M) & (offs_k_cur[None, :] < K)
        weight_mask = (offs_n[None, :] < N) & (offs_k_cur[:, None] < K)

        input_tile = tl.load(input_ptrs, mask=input_mask, other=0.0)
        weight_tile = tl.load(weight_ptrs, mask=weight_mask, other=0.0)

        accumulator += tl.dot(input_tile, weight_tile, allow_tf32=True)

    # Multiply by other (gate/silu) element before writing output
    other_ptrs = other_ptr + offs_m[:, None] * stride_other_m + offs_n[None, :] * stride_other_n
    other_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    other_tile = tl.load(other_ptrs, mask=other_mask, other=0.0)

    output_tile = accumulator * other_tile

    output_ptrs = output_ptr + offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, output_tile, mask=output_mask)


@torch.fx.wrap
def fused_linear_mul_dispatch(*args):
    route = args[-1]
    
    if route == "route_fuse_linear_mul":
        weight, other, input_tensor = args[0], args[1], args[2]
        
        # Reshape input and other to 2D for matmul
        orig_shape = list(input_tensor.shape[:-1]) + [weight.shape[0]]
        
        input_2d = input_tensor.reshape(-1, input_tensor.shape[-1])
        other_2d = other.reshape(-1, other.shape[-1])
        
        M, K = input_2d.shape
        N = weight.shape[0]
        
        output_2d = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
        
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        fused_linear_mul_kernel[grid](
            input_2d, weight, other_2d, output_2d,
            M, N, K,
            input_2d.stride(0), input_2d.stride(1),
            weight.stride(0), weight.stride(1),
            other_2d.stride(0), other_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
        
        return output_2d.reshape(orig_shape)
    
    elif route == "route_broadcast_mul_linear":
        in_0, in_1, in_2, in_3 = args[0], args[1], args[2], args[3]
        
        # mmpose pattern: independent broadcast multiply + linear
        # Compute F.linear(in_3, in_0, None)
        # Compute in_2 * in_1 (broadcast multiply where in_1 is 1D)
        # Return (in_2 * in_1, linear)
        
        linear = torch.nn.functional.linear(in_3, in_0, None)
        
        # Triton broadcast multiply kernel
        N_elements = in_2.numel()
        last_dim = in_2.shape[-1]
        
        result = torch.empty_like(in_2)
        
        # For broadcast multiply, use a simple element-wise kernel
        broadcast_mul_kernel[(triton.cdiv(N_elements, 1024),)](
            in_2, in_1, result, N_elements, last_dim,
            in_1.numel(),
            BLOCK_SIZE=1024,
        )
        
        return (result, linear)
    
    else:
        raise ValueError(f"Unknown route: {route}")


@triton.jit
def broadcast_mul_kernel(
    a_ptr, b_ptr, out_ptr,
    n_elements, last_dim, b_size,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load broadcast b value
    b_offsets = offsets % b_size
    b_vals = tl.load(b_ptr + b_offsets, mask=mask, other=0.0)
    
    # Load a values
    a_vals = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    out_vals = a_vals * b_vals
    
    # Store
    tl.store(out_ptr + offsets, out_vals, mask=mask)


def replacement_func():
    return fused_linear_mul_dispatch