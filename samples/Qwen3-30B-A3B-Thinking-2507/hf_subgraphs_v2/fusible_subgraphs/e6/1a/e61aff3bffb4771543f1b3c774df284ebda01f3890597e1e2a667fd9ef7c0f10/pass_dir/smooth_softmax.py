import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(4, 1, 192)
    tmp_4 = torch.nn.functional.softmax(tmp_3, 2, _stacklevel = 5)
    tmp_5 = tmp_4.unsqueeze(-1)
    return (tmp_5,)

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(-1, 1, -1)
    return (tmp_3,)

# Optimized kernel for softmax
def softmax_kernel(X_ptr, Y_ptr, M, N, BLOCK_M, BLOCK_N):
    # Each program handles a block of M rows
    pid_m = tl.program_id(0)
    start_m = pid_m * BLOCK_M
    end_m = min(start_m + BLOCK_M, M)

    # The row indices we're processing
    offsets_m = start_m + tl.arange(0, BLOCK_M)
    mask_m = offsets_m < M

    # Process the N dimension in blocks
    for start_n in range(0, N, BLOCK_N):
        chunk_n = min(BLOCK_N, N - start_n)
        offsets_n = start_n + tl.arange(0, chunk_n)
        mask_n = offsets_n < N

        # Load data
        X = tl.load(X_ptr + offsets_m[:, None] * N + offsets_n[None, :],
                    mask=mask_m[:, None] & mask_n[None, :],
                    other=-float('inf'))

        # Compute max for this block
        max_val = tl.max(X, axis=1)

        # Compute exponential
        X_exp = tl.exp(X - max_val[:, None])

        # Compute sum of exponentials
        sum_exp = tl.sum(X_exp, axis=1)

        # Compute softmax
        softmax = X_exp / sum_exp[:, None]

        # Store result
        tl.store(Y_ptr + offsets_m[:, None] * N + offsets_n[None, :],
                 softmax,
                 mask=mask_m[:, None] & mask_n[None, :])

# Kernel wrapper
@torch.fx.wrap
def softmax_wrapper(x):
    # x has shape [M, 1, N] -> treat as [M, N]
    M = x.shape[0]
    N = x.shape[2]
    
    # Flatten to 2D tensor
    x_2d = x.view(M, N)
    y_2d = torch.empty_like(x_2d)

    # Tunable block sizes
    BLOCK_M = 128
    BLOCK_N = 64
    
    # Compute grid size
    num_blocks = (M + BLOCK_M - 1) // BLOCK_M
    
    # Launch kernel
    softmax_kernel[(num_blocks,)](
        x_2d,
        y_2d,
        M,
        N,
        BLOCK_M,
        BLOCK_N
    )
    
    # Reshape back to [M, 1, N]
    return y_2d.view(M, 1, N)

# Replacement function
def replacement_func():
    return softmax_wrapper