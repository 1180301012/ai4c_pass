import torch
import triton
import triton.language as tl

# Pattern matching function - must mirror model.py exactly (without cleanup statements)
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return (tmp_4,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# Fused Triton kernel: linear + reshape + softmax
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 19, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 19, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'K'],
)
@triton.jit
def fused_linear_reshape_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K,
    x_stride_m, x_stride_k,
    w_stride_n, w_stride_k,
    b_stride_n,
    out_stride_0, out_stride_1,
    SOFTMAX_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    N: tl.constexpr = NUM_HEADS * SOFTMAX_SIZE

    pid = tl.program_id(0)
    row_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_off < M

    n_idx = tl.arange(0, N)

    # ---- Linear: accumulate x @ w^T + bias ----
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K
        x_ptrs = x_ptr + row_off[:, None] * x_stride_m + k_off[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)
        w_ptrs = w_ptr + n_idx[:, None] * w_stride_n + k_off[None, :] * w_stride_k
        w_vals = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)
        acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=False)

    if b_ptr is not None:
        b_vals = tl.load(b_ptr + n_idx * b_stride_n)
        acc += b_vals[None, :]

    # ---- Softmax over SOFTMAX_SIZE ----
    head_idx = tl.arange(0, NUM_HEADS)

    max_vals = tl.full((BLOCK_M, NUM_HEADS), float('-inf'), dtype=tl.float32)
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        val = acc[:, col]
        max_vals = tl.maximum(max_vals, val)

    sum_vals = tl.zeros((BLOCK_M, NUM_HEADS), dtype=tl.float32)
    exp_cache = []
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        val = acc[:, col]
        e = tl.exp(val - max_vals)
        sum_vals += e
        exp_cache.append(e)

    for pos in range(SOFTMAX_SIZE):
        softmax_val = exp_cache[pos] / sum_vals
        out_row_idx = row_off[:, None] * NUM_HEADS + head_idx[None, :]
        out_ptrs = out_ptr + out_row_idx * out_stride_0 + pos * out_stride_1
        tl.store(out_ptrs, softmax_val, mask=row_mask[:, None])


@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = weight.shape[0]

    SOFTMAX_SIZE = 9
    NUM_HEADS = N // SOFTMAX_SIZE

    out = torch.empty((M * NUM_HEADS, SOFTMAX_SIZE, 1), dtype=x.dtype, device=x.device)

    grid = ((M + 19 - 1) // 19,)

    fused_linear_reshape_softmax_kernel[grid](
        x_ptr=x_2d, w_ptr=weight, b_ptr=bias, out_ptr=out,
        M=M, K=K,
        x_stride_m=x_2d.stride(0), x_stride_k=x_2d.stride(1),
        w_stride_n=weight.stride(0), w_stride_k=weight.stride(1),
        b_stride_n=bias.stride(0),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        SOFTMAX_SIZE=SOFTMAX_SIZE,
        NUM_HEADS=NUM_HEADS,
    )

    return out


def replacement_func():
    return fused_linear_reshape_softmax




    pid = tl.program_id(0)
    row_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_off < M

    n_idx = tl.arange(0, N)

    # ---- Linear computation: accumulate x @ w^T + bias ----
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        # Load x: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + row_off[:, None] * x_stride_m + k_off[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load w: [N, BLOCK_K]
        w_ptrs = w_ptr + n_idx[:, None] * w_stride_n + k_off[None, :] * w_stride_k
        w_vals = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)

        # Accumulate: x_vals [BLOCK_M, BLOCK_K] @ w_vals^T [BLOCK_K, N] -> [BLOCK_M, N]
        acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=False)

    # Add bias: [BLOCK_M, N]
    if b_ptr is not None:
        b_vals = tl.load(b_ptr + n_idx * b_stride_n)
        acc += b_vals[None, :]

    # ---- Softmax ----
    # Reshape [BLOCK_M, N] -> conceptually [BLOCK_M, NUM_HEADS, SOFTMAX_SIZE]
    # For each (row, head), compute softmax over SOFTMAX_SIZE positions
    head_idx = tl.arange(0, NUM_HEADS)

    # Step 1: Find max for each (row, head) group
    max_vals = tl.full((BLOCK_M, NUM_HEADS), float('-inf'), dtype=tl.float32)
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos  # [NUM_HEADS]
        val = acc[:, col]  # [BLOCK_M, NUM_HEADS]
        max_vals = tl.maximum(max_vals, val)

    # Step 2: Compute exp(val - max) and accumulate sum
    sum_vals = tl.zeros((BLOCK_M, NUM_HEADS), dtype=tl.float32)
    exp_cache = []  # cache exp values to avoid recomputation
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        val = acc[:, col]
        e = tl.exp(val - max_vals)
        sum_vals += e
        exp_cache.append(e)

    # Step 3: Normalize and store output
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        softmax_val = exp_cache[pos] / sum_vals  # [BLOCK_M, NUM_HEADS]

        # Output tensor shape: [M * NUM_HEADS, 9, 1]
        # Index for (row, head, pos):
        #   dim0 = row * NUM_HEADS + head
        #   dim1 = pos
        out_row_idx = row_off[:, None] * NUM_HEADS + head_idx[None, :]  # [BLOCK_M, NUM_HEADS]
        out_mask = row_mask[:, None]

        out_ptrs = out_ptr + out_row_idx * out_stride_0 + pos * out_stride_1
        tl.store(out_ptrs, softmax_val, mask=out_mask)


# Kernel wrapper
@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    """
    Fused implementation of: linear(x, weight, bias) -> reshape([-1, 9, 1]) -> softmax(dim=1)

    Input shapes (from weight_meta):
    - bias (in_0): [18]
    - weight (in_1): [18, 128]
    - x (in_2): [1, 19, 128]
    
    Computation:
    - linear output: [1, 19, 18]
    - reshape to [-1, 9, 1]: [38, 9, 1]  (38 = 1*19*18 / 9 = 19*2)
    - softmax over dim=1 (9 elements per group)
    """
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = weight.shape[0]

    SOFTMAX_SIZE = 9
    NUM_HEADS = N // SOFTMAX_SIZE  # = 2

    # Output shape: [M * NUM_HEADS, SOFTMAX_SIZE, 1]
    out = torch.empty((M * NUM_HEADS, SOFTMAX_SIZE, 1), dtype=x.dtype, device=x.device)

    grid = ((M + 19 - 1) // 19,)  # placeholder grid; autotune will override BLOCK_M

    fused_linear_reshape_softmax_kernel[grid](
        x_ptr=x_2d, w_ptr=weight, b_ptr=bias, out_ptr=out,
        M=M, K=K,
        x_stride_m=x_2d.stride(0), x_stride_k=x_2d.stride(1),
        w_stride_n=weight.stride(0), w_stride_k=weight.stride(1),
        b_stride_n=bias.stride(0),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        SOFTMAX_SIZE=SOFTMAX_SIZE,
        NUM_HEADS=NUM_HEADS,
    )

    return out


def replacement_func():
    return fused_linear_reshape_softmax
)
@triton.jit
def fused_linear_reshape_softmax_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K,
    x_stride_m, x_stride_k,
    w_stride_n, w_stride_k,
    b_stride_n,
    out_stride_0, out_stride_1,
    SOFTMAX_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Derived constexpr
    N: tl.constexpr = NUM_HEADS * SOFTMAX_SIZE

    pid = tl.program_id(0)
    row_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_off < M

    n_idx = tl.arange(0, N)

    # ---- Linear computation: accumulate x @ w^T + bias ----
    acc = tl.zeros((BLOCK_M, N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        # Load x: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + row_off[:, None] * x_stride_m + k_off[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Load w: [N, BLOCK_K]
        w_ptrs = w_ptr + n_idx[:, None] * w_stride_n + k_off[None, :] * w_stride_k
        w_vals = tl.load(w_ptrs, mask=k_mask[None, :], other=0.0)

        # Accumulate: x_vals [BLOCK_M, BLOCK_K] @ w_vals^T [BLOCK_K, N] -> [BLOCK_M, N]
        acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=False)

    # Add bias: [BLOCK_M, N]
    if b_ptr is not None:
        b_vals = tl.load(b_ptr + n_idx * b_stride_n)
        acc += b_vals[None, :]

    # ---- Softmax ----
    # Reshape [BLOCK_M, N] -> conceptually [BLOCK_M, NUM_HEADS, SOFTMAX_SIZE]
    # For each (row, head), compute softmax over SOFTMAX_SIZE positions
    head_idx = tl.arange(0, NUM_HEADS)

    # Step 1: Find max for each (row, head) group
    max_vals = tl.full((BLOCK_M, NUM_HEADS), float('-inf'), dtype=tl.float32)
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos  # [NUM_HEADS]
        val = acc[:, col]  # [BLOCK_M, NUM_HEADS]
        max_vals = tl.maximum(max_vals, val)

    # Step 2: Compute exp(val - max) and accumulate sum
    sum_vals = tl.zeros((BLOCK_M, NUM_HEADS), dtype=tl.float32)
    exp_cache = []  # cache exp values to avoid recomputation
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        val = acc[:, col]
        e = tl.exp(val - max_vals)
        sum_vals += e
        exp_cache.append(e)

    # Step 3: Normalize and store output
    for pos in range(SOFTMAX_SIZE):
        col = head_idx * SOFTMAX_SIZE + pos
        softmax_val = exp_cache[pos] / sum_vals  # [BLOCK_M, NUM_HEADS]

        # Output tensor shape: [M * NUM_HEADS, 9, 1]
        # Index for (row, head, pos):
        #   dim0 = row * NUM_HEADS + head
        #   dim1 = pos
        out_row_idx = row_off[:, None] * NUM_HEADS + head_idx[None, :]  # [BLOCK_M, NUM_HEADS]
        out_mask = row_mask[:, None]

        out_ptrs = out_ptr + out_row_idx * out_stride_0 + pos * out_stride_1
        tl.store(out_ptrs, softmax_val, mask=out_mask)


# Kernel wrapper
@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    """
    Fused implementation of: linear(x, weight, bias) -> reshape([-1, 9, 1]) -> softmax(dim=1)

    Input shapes (from weight_meta):
    - bias (in_0): [18]
    - weight (in_1): [18, 128]
    - x (in_2): [1, 19, 128]
    
    Computation:
    - linear output: [1, 19, 18]
    - reshape to [-1, 9, 1]: [38, 9, 1]  (38 = 1*19*18 / 9 = 19*2)
    - softmax over dim=1 (9 elements per group)
    """
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = weight.shape[0]

    SOFTMAX_SIZE = 9
    NUM_HEADS = N // SOFTMAX_SIZE  # = 2

    # Output shape: [M * NUM_HEADS, SOFTMAX_SIZE, 1]
    out = torch.empty((M * NUM_HEADS, SOFTMAX_SIZE, 1), dtype=x.dtype, device=x.device)

    grid = ((M + 19 - 1) // 19,)  # placeholder grid; autotune will override BLOCK_M

    fused_linear_reshape_softmax_kernel[grid](
        x_ptr=x_2d, w_ptr=weight, b_ptr=bias, out_ptr=out,
        M=M, K=K,
        x_stride_m=x_2d.stride(0), x_stride_k=x_2d.stride(1),
        w_stride_n=weight.stride(0), w_stride_k=weight.stride(1),
        b_stride_n=bias.stride(0),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1),
        SOFTMAX_SIZE=SOFTMAX_SIZE,
        NUM_HEADS=NUM_HEADS,
    )

    return out


def replacement_func():
    return fused_linear_reshape_softmax



        
        
    # Each program handles BLOCK_M rows of the linear output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Row offsets
    row_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_off < M

    # Column offsets for the linear output
    col_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    col_mask_n = col_off < N

    # ---- Linear computation: partial dot product ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        k_idx = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load x: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + row_off[:, None] * x_stride_m + k_idx[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)

        # Load w: [BLOCK_N, BLOCK_K]
        w_ptrs = w_ptr + col_off[:, None] * w_stride_n + k_idx[None, :] * w_stride_k
        w_vals = tl.load(w_ptrs, mask=(col_mask_n[:, None] & k_mask[None, :]), other=0.0)

        # Accumulate
        acc += tl.dot(x_vals, tl.trans(w_vals), allow_tf32=False)

    # Add bias
    if b_ptr is not None:
        b_ptrs = b_ptr + col_off * b_stride_n
        b_vals = tl.load(b_ptrs, mask=col_mask_n, other=0.0)
        acc += b_vals[None, :]

    # ---- Softmax on the partial linear results ----
    # The reshape [-1, 9, 1] maps linear output row i, col j to
    # softmax group = (i * N + j) // 9, position = (i * N + j) % 9
    # But since N is a multiple of 9 (N=18, 9*2), for each row i:
    #   group_id within row = col // 9 (i.e., 0 or 1)
    #   global group_id = i * (N // 9) + col // 9
    #   position within group = col % 9

    # For each (row, col) in our block, compute which softmax group it belongs to
    # and its position within that group
    group_in_row = col_off // SOFTMAX_SIZE  # which head within the row
    pos_in_group = col_off % SOFTMAX_SIZE   # position within the softmax group

    # We need to compute softmax across all 9 positions in each group
    # Since BLOCK_N may not cover all 9 positions, we need to do a cross-block reduction

    # Step 1: Compute local max for each (row, group) pair
    # Create a mask for each group within the row
    # For each (row, group), find the max over positions in this block
    max_val = tl.full((BLOCK_M, BLOCK_N), float('-inf'), dtype=tl.float32)

    # For each position in the softmax group, check if it's in our block
    for pos in range(SOFTMAX_SIZE):
        # Which columns in our block correspond to position pos?
        # col = group_in_row * SOFTMAX_SIZE + pos
        # We need: col_off == group_in_row * SOFTMAX_SIZE + pos
        # But group_in_row varies per column... this is complex

        # Alternative approach: iterate over all possible group heads
        pass

    # Alternative simpler approach for small N:
    # Since N=18 and SOFTMAX_SIZE=9, we have exactly 2 groups per row.
    # With BLOCK_N covering enough columns, we can compute softmax directly.

    # Let's use a simpler approach: for each (row, head_idx), compute softmax
    # over all 9 positions

    # Number of softmax groups per row
    num_heads_per_row = N // SOFTMAX_SIZE  # = 2

    # For each row in our block, for each head, compute softmax
    # We iterate over all 9 positions regardless of which are in our block

    # First, compute the max for each (row, head) pair
    # We need the full linear result for all 9 positions, not just the ones in our block
    # Since the linear was only partially computed (just our block of columns),
    # we need a different strategy

    # Strategy: each program computes the FULL linear output for its BLOCK_M rows,
    # but only for the columns in its BLOCK_N. Then we need cross-program
    # communication for softmax, which is hard in Triton.

    # Better strategy: change the grid so that each program handles ALL N columns
    # for its BLOCK_M rows, computing the full linear output and softmax together.
    # This is possible if N is small (N=18).
    # But with BLOCK_N < N, we'd need to loop over N in the kernel.

    # Let me redesign the kernel to loop over N columns entirely within each program.
    pass


# Redesigned kernel: each program handles BLOCK_M rows and ALL N columns
# This allows us to compute softmax within each program since we have all 9 values
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_reshape_softmax_kernel_v2(
    # Pointers
    x_ptr, w_ptr, b_ptr, out_ptr,
    # Dimensions
    M, N, K,
    # Strides for input x: shape [M, K]
    x_stride_m, x_stride_k,
    # Strides for weight w: shape [N, K]
    w_stride_n, w_stride_k,
    # Strides for bias b: shape [N]
    b_stride_n,
    # Strides for output: shape [num_heads, 9, 1]
    out_stride_0, out_stride_1, out_stride_2,
    # Softmax group size
    SOFTMAX_SIZE: tl.constexpr,
    # Number of heads per row
    NUM_HEADS: tl.constexpr,
    # Block size for rows
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    row_off = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = row_off < M

    # For each row, compute the full linear output of size N
    # Then apply reshape + softmax

    # We'll compute the linear output in chunks of BLOCK_K
    # and store the full N-length result in registers
    # Since N is small (18), we can hold it all

    # Accumulate linear result: [BLOCK_M, N]
    n_idx = tl.arange(0, NUM_HEADS * SOFTMAX_SIZE)  # [0, ..., N-1]
    n_mask = n_idx < N

    acc = tl.zeros((BLOCK_M, NUM_HEADS * SOFTMAX_SIZE), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        k_idx = k_off + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        # Load x: [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + row_off[:, None] * x_stride_m + k_idx[None, :] * x_stride_k
        x_vals = tl.load(x_ptrs, mask=(row_mask[:, None] & k_mask[None, :]), other=0.0)

        # Load w: [N, BLOCK_K] - load all N rows
        w_ptrs = w_ptr + n_idx[:, None] * w_stride_n + k_idx[None, :] * w_stride_k
        w_vals = tl.load(w_ptrs, mask=(n_mask[:, None] & k_mask[None, :]), other=0.0)

        # Accumulate: [BLOCK_M, N] += [BLOCK_M, BLOCK_K] @ [BLOCK_K, N]
        # We can't use tl.dot directly since shapes don't match constexpr requirements
        # Use manual outer product accumulation
        # acc += tl.dot(x_vals, tl.trans(w_vals)) won't work because N is not a constexpr multiple
        # Let's use element-wise multiply and sum
        acc += tl.sum(x_vals[:, None, :] * w_vals[None, :, :], axis=2)

    # Add bias: [BLOCK_M, N]
    if b_ptr is not None:
        b_ptrs = b_ptr + n_idx * b_stride_n
        b_vals = tl.load(b_ptrs, mask=n_mask, other=0.0)
        acc += b_vals[None, :]

    # ---- Softmax ----
    # Reshape: [BLOCK_M, N] -> [BLOCK_M, NUM_HEADS, SOFTMAX_SIZE]
    # For each head, softmax over SOFTMAX_SIZE positions

    # Compute max for each (row, head) pair
    # acc layout: [BLOCK_M, NUM_HEADS * SOFTMAX_SIZE]
    # We want to compute max over the SOFTMAX_SIZE dimension

    max_val = tl.full((BLOCK_M, NUM_HEADS), float('-inf'), dtype=tl.float32)
    for pos in range(SOFTMAX_SIZE):
        col_idx = tl.arange(0, NUM_HEADS) * SOFTMAX_SIZE + pos
        val = tl.load(acc_ptr_if_we_had_one, ...)  # This won't work - acc is in registers

    # Since acc is in registers, we can index it directly
    # acc[:, head*9+pos] for each head and pos

    # Actually, let me reconsider. The acc array is [BLOCK_M, N] where N=18.
    # We can manually compute softmax by iterating over the 9 positions.

    # Compute max for each (row, head)
    max_vals = tl.full((BLOCK_M, NUM_HEADS), float('-inf'), dtype=tl.float32)
    for pos in range(SOFTMAX_SIZE):
        # Extract acc[:, head*9 + pos] for all heads
        head_idx = tl.arange(0, NUM_HEADS)
        col_indices = head_idx * SOFTMAX_SIZE + pos
        vals = acc[:, col_indices]  # This should work for register array indexing
        max_vals = tl.maximum(max_vals, vals)

    # Compute exp(val - max) and sum
    sum_vals = tl.zeros((BLOCK_M, NUM_HEADS), dtype=tl.float32)
    exp_results = []  # Store exp values for later
    for pos in range(SOFTMAX_SIZE):
        head_idx = tl.arange(0, NUM_HEADS)
        col_indices = head_idx * SOFTMAX_SIZE + pos
        vals = acc[:, col_indices]
        e = tl.exp(vals - max_vals)
        sum_vals += e
        exp_results.append(e)

    # Normalize
    for pos in range(SOFTMAX_SIZE):
        head_idx = tl.arange(0, NUM_HEADS)
        col_indices = head_idx * SOFTMAX_SIZE + pos
        softmax_val = exp_results[pos] / sum_vals

        # Store to output: shape [num_heads_total, 9, 1]
        # Output index: (row * NUM_HEADS + head) * out_stride_0 + pos * out_stride_1 + 0 * out_stride_2
        # But we need to be careful about the output layout

        # The output tensor after reshape is [-1, 9, 1] where -1 = M * NUM_HEADS
        # So output shape is [M * NUM_HEADS, 9, 1]
        # Output index for (row, head, pos): row * NUM_HEADS + head for dim0, pos for dim1

        out_row = row_off[:, None] * NUM_HEADS + head_idx[None, :]  # [BLOCK_M, NUM_HEADS]
        out_row_mask = row_mask[:, None] & (head_idx[None, :] < NUM_HEADS)

        out_ptrs = out_ptr + out_row * out_stride_0 + pos * out_stride_1
        tl.store(out_ptrs, softmax_val, mask=out_row_mask)


# Kernel wrapper
@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    """
    Fused implementation of: linear(x, weight, bias) -> reshape([-1, 9, 1]) -> softmax(dim=1)
    
    Input shapes:
    - bias: [N] = [18]
    - weight: [N, K] = [18, 128]
    - x: [M, K] = [1*19, 128] (flattened from [1, 19, 128])
    
    Output shape: [M * (N/9), 9, 1] = [38, 9, 1]
    """
    # Flatten x to 2D if needed
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M = x_2d.shape[0]
    K = x_2d.shape[1]
    N = weight.shape[0]

    SOFTMAX_SIZE = 9
    NUM_HEADS = N // SOFTMAX_SIZE  # = 2

    # Output shape after reshape and softmax
    out = torch.empty((M * NUM_HEADS, SOFTMAX_SIZE, 1), dtype=x.dtype, device=x.device)

    BLOCK_M = 4
    BLOCK_K = 64

    grid = ((M + BLOCK_M - 1) // BLOCK_M,)

    fused_linear_reshape_softmax_kernel_v2[grid](
        x_ptr=x_2d, w_ptr=weight, b_ptr=bias, out_ptr=out,
        M=M, N=N, K=K,
        x_stride_m=x_2d.stride(0), x_stride_k=x_2d.stride(1),
        w_stride_n=weight.stride(0), w_stride_k=weight.stride(1),
        b_stride_n=bias.stride(0),
        out_stride_0=out.stride(0), out_stride_1=out.stride(1), out_stride_2=out.stride(2),
        SOFTMAX_SIZE=SOFTMAX_SIZE,
        NUM_HEADS=NUM_HEADS,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
    )

    return out

def replacement_func():
    return fused_linear_reshape_softmax