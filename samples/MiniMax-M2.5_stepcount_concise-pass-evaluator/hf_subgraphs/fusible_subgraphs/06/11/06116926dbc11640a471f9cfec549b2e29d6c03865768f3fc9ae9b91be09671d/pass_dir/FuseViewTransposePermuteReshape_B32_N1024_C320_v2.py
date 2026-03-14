import torch
import triton
import triton.language as tl


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
    num_elements = B * C * H * W
    elements_per_pid = BLOCK_SIZE
    start_pid = pid * elements_per_pid
    
    for i in range(elements_per_pid):
        idx = start_pid + i
        mask = idx < num_elements
        
        b = idx // (C * H * W)
        remainder = idx % (C * H * W)
        c = remainder // (H * W)
        remainder = remainder % (H * W)
        h = remainder // W
        w = remainder % W
        
        n_idx = h * W + w
        input_idx = b * N * C + n_idx * C + c
        
        val = tl.load(in_ptr + input_idx, mask=mask)
        tl.store(out_ptr + idx, val, mask=mask)


@triton.jit
def fused_view_transpose_kernel(
    in_ptr,
    out_ptr,
    B: tl.constexpr,
    N: tl.constexpr,
    C: tl.constexpr,
    num_groups: tl.constexpr,
    group_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: view + transpose for pattern view(B, -1, k, 64) -> transpose(1,2)"""
    pid = tl.program_id(0)
    num_elements = B * num_groups * group_size * C
    elements_per_pid = BLOCK_SIZE
    start_pid = pid * elements_per_pid
    
    for i in range(elements_per_pid):
        idx = start_pid + i
        mask = idx < num_elements
        
        # Output is [B, group_size, num_groups, C]
        b = idx // (num_groups * group_size * C)
        remainder = idx % (num_groups * group_size * C)
        g = remainder // (group_size * C)
        remainder = remainder % (group_size * C)
        s = remainder // C
        c = remainder % C
        
        # Input is [B, N, C] = [B, num_groups * group_size, C]
        # view: [B, num_groups, group_size, C]
        # transpose(1,2): [B, group_size, num_groups, C]
        n_idx = g * group_size + s
        input_idx = b * N * C + n_idx * C + c
        
        val = tl.load(in_ptr + input_idx, mask=mask)
        tl.store(out_ptr + idx, val, mask=mask)


@torch.fx.wrap
def fused_permute_reshape(x, B, N, C, H, W):
    """Fused permute and reshape operation."""
    out = torch.empty((B, C, H, W), dtype=x.dtype, device=x.device)
    num_elements = B * C * H * W
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_permute_reshape_kernel[(num_programs,)](
        x, out, B, N, C, H, W, BLOCK_SIZE
    )
    return out


@torch.fx.wrap
def fused_view_transpose(x, B, N, C, num_groups, group_size):
    """Fused view and transpose operation."""
    out_shape = (B, group_size, num_groups, C)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    num_elements = B * num_groups * group_size * C
    BLOCK_SIZE = 1024
    num_programs = (num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_view_transpose_kernel[(num_programs,)](
        x, out, B, N, C, num_groups, group_size, BLOCK_SIZE
    )
    return out


def pattern(in_0, in_1):
    """
    Pattern: view -> transpose on in_1, permute -> reshape on in_0
    Graph: face-parsing_start422_end426_52 (fusible_subgraphs/7)
    - in_0: [32, 1024, 320] -> permute(0, 2, 1) -> [32, 320, 1024] -> reshape -> [32, 320, 32, 32]
    - in_1: [32, 1024, 320] -> view(32, -1, 5, 64) -> [32, 32, 5, 64] -> transpose(1, 2) -> [32, 5, 32, 64]
    """
    # in_1 path: view -> transpose
    tmp_0 = in_1.view(32, -1, 5, 64)
    tmp_1 = tmp_0.transpose(1, 2)
    
    # in_0 path: permute -> reshape
    tmp_2 = in_0.permute(0, 2, 1)
    tmp_3 = tmp_2.reshape(32, 320, 32, 32)
    
    return tmp_1, tmp_3


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    def replacement(in_0, in_1):
        B = 32
        N = 1024
        C = 320
        H = 32
        W = 32
        
        # For view(32, -1, 5, 64) with input [32, 1024, 320]:
        # The -1 computes to: N * C / (5 * 64) = 1024 * 320 / 320 = 1024
        # After view: [32, 1024, 5, 64]
        # After transpose(1,2): [32, 5, 1024, 64]
        # So num_groups = 1024, group_size = 5
        num_groups = 1024
        group_size = 5
        
        tmp_1 = fused_view_transpose(in_1, B, N, C, num_groups, group_size)
        tmp_3 = fused_permute_reshape(in_0, B, N, C, H, W)
        
        return tmp_1, tmp_3
    
    return replacement