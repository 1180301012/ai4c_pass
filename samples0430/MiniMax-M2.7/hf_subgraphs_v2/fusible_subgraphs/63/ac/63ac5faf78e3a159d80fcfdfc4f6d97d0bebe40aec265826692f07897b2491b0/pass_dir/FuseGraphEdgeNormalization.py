import torch
import triton
import triton.language as tl
from torch import inf


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Match the graph edge normalization pattern with linear layer.
    
    Pattern:
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_2[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_2[in_2]
    tmp_8 = tmp_6 * tmp_7
    linear = torch.nn.functional.linear(in_0, in_1, None)
    """
    tmp_2 = in_3.pow_(-0.5)
    tmp_3 = tmp_2.__eq__(inf)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 0)
    tmp_5 = tmp_4[in_5]
    tmp_6 = tmp_5 * in_4
    tmp_7 = tmp_4[in_2]
    tmp_8 = tmp_6 * tmp_7
    linear = torch.nn.functional.linear(in_0, in_1, None)
    return tmp_8, linear


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_edges'],
)
@triton.jit
def graph_edge_norm_kernel(
    deg_ptr, row_ptr, col_ptr, edge_weight_ptr, out_ptr,
    n_deg, n_edges, BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate block range
    start = pid * BLOCK_SIZE
    end = start + BLOCK_SIZE
    
    # Create mask for valid indices
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_edges
    
    # Load row, col, and edge_weight indices
    row = tl.load(row_ptr + start + offsets, mask=mask, other=0)
    col = tl.load(col_ptr + start + offsets, mask=mask, other=0)
    edge_w = tl.load(edge_weight_ptr + start + offsets, mask=mask, other=1.0)
    
    # Load degree values and compute deg^(-0.5)
    deg_row = tl.load(deg_ptr + row, mask=mask, other=0.0)
    deg_col = tl.load(deg_ptr + col, mask=mask, other=0.0)
    
    # Compute normalized values: deg^(-0.5), replace inf with 0
    inv_sqrt_deg_row = 1.0 / tl.sqrt(deg_row + 1e-8)
    inv_sqrt_deg_col = 1.0 / tl.sqrt(deg_col + 1e-8)
    
    # Handle inf values (set to 0)
    inv_sqrt_deg_row = tl.where(tl.isinf(inv_sqrt_deg_row), 0.0, inv_sqrt_deg_row)
    inv_sqrt_deg_col = tl.where(tl.isinf(inv_sqrt_deg_col), 0.0, inv_sqrt_deg_col)
    
    # Compute final normalized edge weight: deg_norm[row] * deg_norm[col] * edge_weight
    out = inv_sqrt_deg_row * inv_sqrt_deg_col * edge_w
    
    # Store results
    tl.store(out_ptr + start + offsets, out, mask=mask)


@torch.fx.wrap
def triton_graph_edge_norm(deg, row, col, edge_weight):
    """
    Optimized graph edge normalization: deg^(-0.5) with inf handling,
    then compute deg_norm[row] * deg_norm[col] * edge_weight
    """
    n_edges = row.shape[0]
    n_deg = deg.shape[0]
    
    # Use 1D grid with autotuned BLOCK_SIZE
    BLOCK_SIZE = 1024  # Will be overridden by autotune
    num_programs = (n_edges + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    out = torch.empty(n_edges, dtype=deg.dtype, device=deg.device)
    
    # Launch kernel
    graph_edge_norm_kernel[(num_programs,)](
        deg_ptr=deg,
        row_ptr=row,
        col_ptr=col,
        edge_weight_ptr=edge_weight,
        out_ptr=out,
        n_deg=n_deg,
        n_edges=n_edges,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def triton_matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    """
    Triton kernel for matrix multiplication: C = A @ B
    """
    # Get program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Calculate block start positions
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create masks for valid indices
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Initialize accumulator
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        # Load A block
        a_offs = offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak
        a_mask = mask_m[:, None] & ((k + offs_k)[None, :] < K)
        a = tl.load(a_ptr + a_offs, mask=a_mask, other=0.0)
        
        # Load B block
        b_offs = (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = ((k + offs_k)[:, None] < K) & mask_n[None, :]
        b = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)
        
        # Compute block multiplication and accumulate
        accumulator += tl.dot(a, b)
        
        # Advance K offset
        offs_k = offs_k + BLOCK_K
    
    # Write result
    c_offs = offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(c_ptr + c_offs, accumulator, mask=c_mask)


@torch.fx.wrap
def triton_matmul(input_tensor, weight):
    """
    Triton matrix multiplication: output = input @ weight.T
    """
    M = input_tensor.shape[0]
    K = input_tensor.shape[1]
    N = weight.shape[0]  # Weight is (N, K), transposed to (K, N) for matmul
    
    # Allocate output
    output = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Define block sizes
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 32
    
    # Calculate grid dimensions
    grid_m = (M + BLOCK_M - 1) // BLOCK_M
    grid_n = (N + BLOCK_N - 1) // BLOCK_N
    
    # Launch kernel
    triton_matmul_kernel[(grid_m, grid_n)](
        a_ptr=input_tensor,
        b_ptr=weight.t(),  # Transpose weight for correct matmul: input @ weight.t()
        c_ptr=output,
        M=M, N=N, K=K,
        stride_am=input_tensor.stride(0), stride_ak=input_tensor.stride(1),
        stride_bk=weight.t().stride(0), stride_bn=weight.t().stride(1),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    
    return output


def replacement_func():
    """
    Returns a function that performs the fused graph edge normalization + linear.
    """
    return fused_graph_linear


def fused_graph_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Fused function for graph edge normalization and linear layer.
    """
    # Use optimized Triton kernel for graph edge computation
    tmp_8 = triton_graph_edge_norm(in_3, in_5, in_2, in_4)
    
    # Use Triton matmul for linear layer (equivalent to linear(input, weight, None))
    linear = triton_matmul(in_0, in_1)
    
    return tmp_8, linear