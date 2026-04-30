import torch
import triton
import triton.language as tl

def _linear_opt(input_tensor, weight, bias=None):
    """
    Optimized linear layer using Triton.
    Uses torch.empty for output allocation (allowed API).
    """
    original_shape = input_tensor.shape
    in_features = original_shape[-1]
    out_features = weight.shape[0]
    
    if len(original_shape) == 3:
        M1, M2, K = original_shape
        stride_in_0 = M2 * K
        stride_in_1 = K
        output_shape = (M1, M2, out_features)
        stride_out_0 = M2 * out_features
        stride_out_1 = out_features
        M = M1 * M2
    else:
        M = 1
        for dim in original_shape[:-1]:
            M *= dim
        M2 = 1
        stride_in_0 = original_shape[-1]
        stride_in_1 = 1
        output_shape = original_shape[:-1] + (out_features,)
        stride_out_0 = out_features
        stride_out_1 = 1
    
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_N = 512
    BLOCK_K = 32
    grid_m = M
    
    _linear_kernel[(grid_m,)](
        input_ptr=input_tensor,
        weight_ptr=weight,
        bias_ptr=bias if bias is not None else 0,
        output_ptr=output,
        M=M,
        M2=M2,
        N=out_features,
        K=in_features,
        stride_in_0=stride_in_0,
        stride_in_1=stride_in_1,
        stride_out_0=stride_out_0,
        stride_out_1=stride_out_1,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output


@triton.jit
def _linear_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    M: tl.constexpr,
    M2: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_in_0: tl.constexpr,
    stride_in_1: tl.constexpr,
    stride_out_0: tl.constexpr,
    stride_out_1: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    m2 = pid % M2
    m1 = pid // M2
    n_idx = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_idx = k + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K
        n_mask = n_idx < N
        in_offset = m1 * stride_in_0 + m2 * stride_in_1 + k_idx
        in_ptrs = input_ptr + in_offset
        x = tl.load(in_ptrs, mask=k_mask, other=0.0)
        w_ptrs = weight_ptr + k_idx[:, None] * N + n_idx[None, :]
        w = tl.load(w_ptrs, mask=k_mask[:, None] & n_mask[None, :], other=0.0)
        acc += tl.sum(x[:, None] * w, axis=0)
    
    if bias_ptr:
        bias = tl.load(bias_ptr + n_idx, mask=n_idx < N, other=0.0)
        acc += bias
    
    out_offset = m1 * stride_out_0 + m2 * stride_out_1 + n_idx
    out_ptrs = output_ptr + out_offset
    out_mask = n_idx < N
    tl.store(out_ptrs, acc, mask=out_mask)


def _view_transpose_opt(in_tensor):
    """
    Optimized view + transpose + contiguous using Triton.
    """
    batch, seq, hidden = in_tensor.shape
    head_dim = 64
    heads = hidden // head_dim
    output_shape = (batch, heads, seq, head_dim)
    output = torch.empty(output_shape, dtype=in_tensor.dtype, device=in_tensor.device)
    
    total_elements = batch * heads * seq * head_dim
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _view_transpose_kernel[(num_programs,)](
        input_ptr=in_tensor,
        output_ptr=output,
        batch=batch,
        seq=seq,
        heads=heads,
        head_dim=head_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output


@triton.jit
def _view_transpose_kernel(
    input_ptr,
    output_ptr,
    batch: tl.constexpr,
    seq: tl.constexpr,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * heads * seq * head_dim
    
    if pid >= total_elements:
        return
    
    remaining = pid
    out_batch = remaining // (heads * seq * head_dim)
    remaining = remaining % (heads * seq * head_dim)
    out_heads = remaining // (seq * head_dim)
    remaining = remaining % (seq * head_dim)
    out_seq = remaining // head_dim
    out_dim = remaining % head_dim
    
    in_hidden = out_heads * head_dim + out_dim
    in_idx = out_batch * seq * heads * head_dim + out_seq * heads * head_dim + in_hidden
    
    value = tl.load(input_ptr + in_idx)
    tl.store(output_ptr + pid, value)


@torch.fx.wrap
def _dispatch_wrapper(*args, route=""):
    """
    Shared dispatch wrapper for all optimized operations.
    Routes to the appropriate implementation based on route string.
    """
    # Extract route from the last element if not passed as keyword
    if not route and len(args) > 0:
        route = args[-1]
        args = args[:-1]
    
    if route == "linear":
        return _linear_opt(args[0], args[1], args[2])
    elif route == "view_transpose":
        return _view_transpose_opt(args[0])
    else:
        # Fallback - shouldn't happen
        raise ValueError(f"Unknown route: {route}")


def pattern(in_0, in_1, in_2):
    """
    Match: torch.nn.functional.linear(input, weight, bias)
    """
    result = torch.nn.functional.linear(in_2, in_1, in_0)
    return result


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "linear")


def replacement_func():
    return _dispatch_wrapper