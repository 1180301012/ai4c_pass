import torch
import triton
import triton.language as tl

# ============================================================
# Shared Triton Kernel Library
# ============================================================

@triton.jit
def fused_linear_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    B, S, D_in, D_out,
    BLOCK_SIZE_K: tl.constexpr,
):
    """Fused linear kernel"""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    if pid_b >= B or pid_d >= D_out:
        return
    
    weight_offsets = pid_d * D_in + tl.arange(0, BLOCK_SIZE_K)
    weight_mask = weight_offsets < D_out * D_in
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for s in range(S):
        w = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        input_offsets = pid_b * S * D_in + s * D_in + tl.arange(0, BLOCK_SIZE_K)
        input_mask = input_offsets < B * S * D_in
        inp = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        acc += tl.sum(inp * w)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_d)
        acc = acc + bias
    
    # Store to [B, S, D_out] layout
    for s in range(S):
        offset = pid_b * S * D_out + s * D_out + pid_d
        result = acc.to(tl.float16)
        tl.store(output_ptr + offset, result)


@triton.jit
def fused_linear_dropout01_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    B, S, D_in, D_out,
    BLOCK_SIZE_K: tl.constexpr,
    seed: tl.constexpr,
):
    """Fused linear + dropout(0.1) kernel"""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    if pid_b >= B or pid_d >= D_out:
        return
    
    weight_offsets = pid_d * D_in + tl.arange(0, BLOCK_SIZE_K)
    weight_mask = weight_offsets < D_out * D_in
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for s in range(S):
        w = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        input_offsets = pid_b * S * D_in + s * D_in + tl.arange(0, BLOCK_SIZE_K)
        input_mask = input_offsets < B * S * D_in
        inp = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        acc += tl.sum(inp * w)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_d)
        acc = acc + bias
    
    # Apply dropout p=0.1
    rng_offset = seed + pid_b * 1000 + pid_d
    random = tl.rand(rng_offset, 0.0)
    mask = random > 0.1
    result = tl.where(mask, acc * 1.111111, 0.0)
    
    # Store to [B, S, D_out] layout
    for s in range(S):
        offset = pid_b * S * D_out + s * D_out + pid_d
        result_f16 = result.to(tl.float16)
        tl.store(output_ptr + offset, result_f16)


@triton.jit
def fused_linear_dropout005_kernel(
    input_ptr, weight_ptr, bias_ptr,
    output_ptr,
    B, S, D_in, D_out,
    BLOCK_SIZE_K: tl.constexpr,
    seed: tl.constexpr,
):
    """Fused linear + dropout(0.05) kernel for bfloat16"""
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    if pid_b >= B or pid_d >= D_out:
        return
    
    weight_offsets = pid_d * D_in + tl.arange(0, BLOCK_SIZE_K)
    weight_mask = weight_offsets < D_out * D_in
    
    acc = tl.zeros((1,), dtype=tl.float32)
    
    for s in range(S):
        w = tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0)
        input_offsets = pid_b * S * D_in + s * D_in + tl.arange(0, BLOCK_SIZE_K)
        input_mask = input_offsets < B * S * D_in
        inp = tl.load(input_ptr + input_offsets, mask=input_mask, other=0.0)
        acc += tl.sum(inp * w)
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + pid_d)
        acc = acc + bias
    
    # Apply dropout p=0.05
    rng_offset = seed + pid_b * 1000 + pid_d
    random = tl.rand(rng_offset, 0.0)
    mask = random > 0.05
    result = tl.where(mask, acc * 1.0526316, 0.0)
    
    # Store to [B, S, D_out] layout - keep float32 for bfloat16
    for s in range(S):
        offset = pid_b * S * D_out + s * D_out + pid_d
        result_f32 = result.to(tl.float32)
        tl.store(output_ptr + offset, result_f32)


# ============================================================
# Wrapper Functions
# ============================================================

@torch.fx.wrap
def fused_linear_wrapper(in_0, in_1, in_2, route):
    """Shared wrapper that routes to the correct kernel based on route string"""
    B, S, D_in = in_2.shape
    D_out = in_1.shape[0]
    
    output = torch.empty(B, S, D_out, dtype=in_2.dtype, device=in_2.device)
    
    grid = (B, D_out)
    BLOCK_SIZE_K = min(32, D_in)
    seed = 12345
    
    if route == "dropout01":
        fused_linear_dropout01_kernel[grid](
            in_2, in_1, in_0, output,
            B, S, D_in, D_out, BLOCK_SIZE_K, seed,
        )
    elif route == "dropout005":
        fused_linear_dropout005_kernel[grid](
            in_2, in_1, in_0, output,
            B, S, D_in, D_out, BLOCK_SIZE_K, seed,
        )
    else:  # dropout0 or default
        fused_linear_kernel[grid](
            in_2, in_1, in_0, output,
            B, S, D_in, D_out, BLOCK_SIZE_K,
        )
    
    # Create transpose output
    output_transpose = output.transpose(1, 2)
    
    return output, output_transpose


@torch.fx.wrap
def fused_linear_wrapper_r43(in_0, in_1, in_2, route):
    """Shared wrapper for (transpose, dropout) return order"""
    B, S, D_in = in_2.shape
    D_out = in_1.shape[0]
    
    output = torch.empty(B, S, D_out, dtype=in_2.dtype, device=in_2.device)
    
    grid = (B, D_out)
    BLOCK_SIZE_K = min(32, D_in)
    seed = 12345
    
    # All use the same kernel for p=0.0
    fused_linear_kernel[grid](
        in_2, in_1, in_0, output,
        B, S, D_in, D_out, BLOCK_SIZE_K,
    )
    
    output_transpose = output.transpose(1, 2)
    
    # Return order: (transpose, dropout)
    return output_transpose, output


# ============================================================
# Pass 1: dropout(0.1) + return (tmp_3, tmp_4)
# ============================================================

def pattern_p01(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.1, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args_p01(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dropout01")


def replacement_func_p01():
    return fused_linear_wrapper


# ============================================================
# Pass 2: dropout(0.05) + return (tmp_3, tmp_4)
# ============================================================

def pattern_p005(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.05, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args_p005(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dropout005")


def replacement_func_p005():
    return fused_linear_wrapper


# ============================================================
# Pass 3: dropout(0.0) + return (tmp_3, tmp_4)
# ============================================================

def pattern_p0_34(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_3, tmp_4


def replacement_args_p0_34(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dropout0")


def replacement_func_p0_34():
    return fused_linear_wrapper


# ============================================================
# Pass 4: dropout(0.0) + return (tmp_4, tmp_3)
# ============================================================

def pattern_p0_43(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.nn.functional.dropout(linear, 0.0, False, False)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4, tmp_3


def replacement_args_p0_43(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "dropout0")


def replacement_func_p0_43():
    return fused_linear_wrapper_r43