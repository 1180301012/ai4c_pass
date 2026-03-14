import torch
import triton
import triton.language as tl


@triton.jit
def fused_view_transpose_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: view(B, N, C) -> view(B, N/k, k, C) -> transpose(B, N/k, k, C) -> (B, k, N/k, C)"""
    pid = tl.program_id(0)
    num_elements = B * H * W * C  # Total elements in output
    elements_per_pid = BLOCK_SIZE
    start_pid = pid * elements_per_pid
    
    # Calculate offsets
    for i in range(elements_per_pid):
        idx = start_pid + i
        if idx >= num_elements:
            break
        
        # Output is [B, H, W, C] - need to map back to input [B, N, C]
        # For the transpose output, the final shape is [B, 1, H, C] or [B, k, N/k, C]
        # Let's compute the output indices
        b = idx // (H * W * C)
        remainder = idx % (H * W * C)
        c = remainder // (H * W)
        remainder = remainder % (H * W)
        h = remainder // W
        w = remainder % W
        
        # Output is at [b, h, w, c] but we need to compute based on the pattern
        # view(B, -1, k, C) with k=1: [B, N/1, 1, C] = [B, N, 1, C]
        # transpose(1, 2): [B, 1, N, C]
        # Input is [B, N, C], so index is [b, h*W + w, c]
        n_idx = h * W + w
        input_idx = b * N * C + n_idx * C + c
        
        val = tl.load(in_ptr + input_idx)
        tl.store(out_ptr + idx, val)


@triton.jit
def fused_permute_reshape_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: permute(0, 2, 1) + reshape(B, C, N) -> (B, C, H, W)"""
    pid = tl.program_id(0)
    num_elements = B * C * H * W  # Total elements in output
    elements_per_pid = BLOCK_SIZE
    start_pid = pid * elements_per_pid
    
    for i in range(elements_per_pid):
        idx = start_pid + i
        if idx >= num_elements:
            break
        
        # Output is [B, C, H, W]
        b = idx // (C * H * W)
        remainder = idx % (C * H * W)
        c = remainder // (H * W)
        remainder = remainder % (H * W)
        h = remainder // W
        w = remainder % W
        
        # Input is [B, N, C], permute(0, 2, 1) gives [B, C, N]
        # reshape to [B, C, H, W] where H*W = N
        n_idx = h * W + w
        input_idx = b * N * C + n_idx * C + c
        
        val = tl.load(in_ptr + input_idx)
        tl.store(out_ptr + idx, val)


@torch.fx.wrap
def fused_view_transpose(x, B, num_groups, group_size, C):
    """Fused view and transpose operation."""
    # view(B, -1, k, C) where k=1: [B, N/k, k, C] = [B, N, 1, C]
    # transpose(1, 2): [B, k, N/k, C] = [B, 1, N/k, C]
    N = B * num_groups * group_size * C
    H = num_groups * group_size
    W = 1
    
    out_shape = (B, 1, num_groups * group_size, C)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    num_elements = B * H * W * C
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_view_transpose_kernel[(num_programs,)](
        x, out, B, N, C, H, W, BLOCK_SIZE
    )
    return out


@torch.fx.wrap
def fused_permute_reshape(x, B, N, C, H, W):
    """Fused permute and reshape operation."""
    # permute(0, 2, 1): [B, N, C] -> [B, C, N]
    # reshape: [B, C, N] -> [B, C, H, W]
    out = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
    
    num_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_permute_reshape_kernel[(num_programs,)](
        x, out, B, N, C, H, W, BLOCK_SIZE
    )
    return out


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def pattern(in_0, in_1):
    """
    Pattern: view -> transpose on in_1, permute -> reshape on in_0
    For graph 7/face-parsing_start46_end50_12:
      in_0: [32, 16384, 64] -> permute(0, 2, 1) -> [32, 64, 16384] -> reshape -> [32, 64, 128, 128]
      in_1: [32, 16384, 64] -> view(32, -1, 1, 64) -> [32, 256, 1, 64] -> transpose(1, 2) -> [32, 1, 256, 64]
    Returns: (tmp_1, tmp_3) as in the original model
    """
    # in_1 path: view -> transpose
    # k = 1 case (C=64)
    tmp_0 = in_1.view(32, -1, 1, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    
    # in_0 path: permute -> reshape
    # H * W = N = 16384 = 128 * 128
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 64, 128, 128)
    
    return tmp_1, tmp_3


def replacement_func():
    def replacement(in_0, in_1):
        B = in_0.shape[0]
        N = in_0.shape[1]
        C = in_0.shape[2]
        
        # Determine H, W from N (find factors to get H*W = N)
        # Try to find sqrt(N) as both H and W for square-like shape
        import math
        H = int(math.sqrt(N))
        while N % H != 0:
            H -= 1
        W = N // H
        
        # in_1: view -> transpose
        # view(32, -1, k, 64) where k is the inner dim from view
        # The pattern uses view(..., 1, 64) or view(..., 5, 64)
        # k is determined by C // 64 for k=1 case or inferred from reshape
        # For C=64, k=1; for C=320, k=5; for C=160, k=5
        group_size = C // 64  # This matches the 64 in view(32, -1, k, 64)
        num_groups = N // group_size
        
        tmp_1 = fused_view_transpose(in_1, B, num_groups, group_size, C)
        
        # in_0: permute -> reshape
        # permute(0, 2, 1): [B, N, C] -> [B, C, N]
        # reshape: [B, C, N] -> [B, C, H, W]
        tmp_3 = fused_permute_reshape(in_0, B, N, C, H, W)
        
        return tmp_1, tmp_3
    
    return replacement