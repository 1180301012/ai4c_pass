import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 256}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_K': 512}, num_stages=4, num_warps=8),
    ],
    key=['B', 'M', 'K'],
)
@triton.jit
def fused_mul_add_unbind_permute_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_0_ptr, out_1_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for:
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0], tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    
    Input shapes:
    - in_0: [2, K] = [2, 128]
    - in_1: [1, 1, 2, K] = [1, 1, 2, 128]
    - in_2: [B, M, 1, K] = [B, 17, 1, 128]
    
    Output shapes:
    - out_0 (tmp_6): [B, K, M] = [B, 128, 17]
    - out_1 (tmp_4): [B, M, K] = [B, 17, 128]
    """
    # Get program ID for batch dimension
    batch_idx = tl.program_id(0)
    
    # Calculate offset for batch
    batch_offset = batch_idx * M * N * K
    
    # Each program processes one element
    # We process in a blocked manner
    offs = tl.arange(0, BLOCK_SIZE)
    
    # Load in_0 [2, K] - need to broadcast across B and M
    # in_0 shape is [2, K], indexed by tmp_2[..., idx, :]
    # We need in_0[idx, :] where idx is 0 or 1
    
    # Pre-load in_0 for both indices
    in_0_ptr_0 = in_0_ptr  # shape [2, K], row 0
    in_0_ptr_1 = in_0_ptr + K  # shape [2, K], row 1
    
    # Loop over M (17) and N (2) dimensions
    for m in range(M):
        for n in range(N):
            # Calculate output offsets
            # out_0 is [B, K, M] = [B, 128, 17], indexed as [batch, k, m]
            # out_1 is [B, M, K] = [B, 17, 128], indexed as [batch, m, k]
            
            out_0_offset = batch_offset + m * K + n * K * M  # Wrong
            out_0_offset = batch_offset + n * K * M + m * K  # [batch, k, m] = batch * K * M + k * M + m
            
            out_1_offset = batch_offset + m * K + n  # [batch, m, k] = batch * M * K + m * K + k
            
            # Load in_1: [B, M, 1, K] at [batch, m, 0, :]
            in_1_base = batch_idx * M * 1 * K + m * 1 * K + 0 * K
            # Load in_2: [B, M, 1, K] at [batch, m, 0, :]
            in_2_base = batch_idx * M * 1 * K + m * 1 * K + 0 * K
            
            # Now compute for each k
            for k in range(K):
                # Load in_1[b, m, 0, k]
                in_1_val = tl.load(in_1_ptr + in_1_base + k)
                # Load in_2[b, m, 0, k]
                in_2_val = tl.load(in_2_ptr + in_2_base + k)
                
                # tmp_1 = in_2 * in_1
                tmp_1_val = in_2_val * in_1_val
                
                # tmp_2 = tmp_1 + in_0[n, k]
                if n == 0:
                    in_0_val = tl.load(in_0_ptr + k)
                else:
                    in_0_val = tl.load(in_0_ptr + K + k)
                
                tmp_2_val = tmp_1_val + in_0_val
                
                # Store to out_1 (tmp_4 = tmp_3[0], tmp_5 = tmp_3[1])
                # tmp_3[0] goes to tmp_4 (out_1), tmp_3[1] goes to tmp_5 which then becomes tmp_6
                # When n=0: tmp_2[..., 0, :] -> tmp_4 = out_1 [batch, m, k]
                # When n=1: tmp_2[..., 1, :] -> tmp_5 -> permute -> out_0 [batch, k, m]
                
                if n == 0:
                    # Store to out_1 [batch, m, k]
                    tl.store(out_1_ptr + out_1_offset + k, tmp_2_val)
                else:
                    # Store to out_0 [batch, k, m]
                    # tmp_6 = tmp_5.permute(0, 2, 1)
                    # tmp_5 has shape [B, M, K], permute(0, 2, 1) gives [B, K, M]
                    # So out_0[batch, k, m] = tmp_2_val when n=1
                    out_0_offset_n1 = batch_offset + k * M + m
                    tl.store(out_0_ptr + out_0_offset_n1, tmp_2_val)


# Better approach: use a simpler parallelization
@triton.jit
def fused_mul_add_unbind_permute_kernel_v2(
    in_0_ptr, in_1_ptr, in_2_ptr,
    out_0_ptr, out_1_ptr,
    B: tl.constexpr, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_in_0: tl.constexpr,
    stride_in_1_m: tl.constexpr, stride_in_1_k: tl.constexpr,
    stride_in_2_m: tl.constexpr, stride_in_2_k: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Fused kernel for:
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + in_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0], tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    
    Input shapes:
    - in_0: [2, K] = [2, 128]
    - in_1: [1, 1, 2, K] = [1, 1, 2, 128] 
    - in_2: [B, M, 1, K] = [B, 17, 1, 128]
    
    After broadcasting:
    tmp_1 = in_2 * in_1: [B, M, 1, K] * [1, 1, 2, K] = [B, M, 2, K]
    tmp_2 = tmp_1 + in_0: [B, M, 2, K] + [2, K] = [B, M, 2, K]
    tmp_3 = unbind(tmp_2, dim=2) -> [B, M, K] and [B, M, K]
    tmp_6 = tmp_5.permute(0, 2, 1) -> [B, K, M]
    
    Output shapes:
    - out_0 (tmp_6): [B, K, M] = [B, 128, 17]
    - out_1 (tmp_4): [B, M, K] = [B, 17, 128]
    """
    # Get program ID - we parallelize over B * M * K elements
    pid = tl.program_id(0)
    
    # Calculate b, m, k indices
    # Total elements per batch = M * K
    elements_per_batch = M * K
    b = pid // elements_per_batch
    remainder = pid % elements_per_batch
    m = remainder // K
    k = remainder % K
    
    # Bounds check
    if b >= B:
        return
    
    # Calculate offsets
    # in_1: [1, 1, 2, K] -> stride [1, 2, 2*K, K] = [0, 0, K, 1] actually
    # Let's think about the strides more carefully
    # in_1 is [1, 1, 2, K], but let's view it as [B, M, N, K] with B=1, M=1
    # Actually in_1 has shape [1, 1, 2, K], but the broadcast happens such that
    # in_1[..., n, :] is used for all b, m
    
    # Since in_1 is [1, 1, 2, K], the effective stride for dim 0, 1 is 0 (broadcast)
    # in_1[n, k] is at offset n*K + k
    
    # in_2: [B, M, 1, K]
    # in_2[b, m, 0, k] -> offset b * M * 1 * K + m * 1 * K + 0 * K + k
    # = b * M * K + m * K + k
    
    # in_0: [2, K]
    # in_0[n, k] -> offset n * K + k
    
    # Compute offsets
    in_1_offset = 0  # in_1 is [1,1,2,K], indexed as [0, 0, n, k] = n*K + k
    in_2_offset = b * stride_in_2_m * K + m * K + k  # [B, M, 1, K]
    in_0_offset = k  # we'll load based on n
    
    # For output:
    # out_0: [B, K, M], indexed as [b, k, m] -> offset b*K*M + k*M + m
    out_0_offset = b * K * M + k * M + m
    
    # out_1: [B, M, K], indexed as [b, m, k] -> offset b*M*K + m*K + k  
    out_1_offset = b * M * K + m * K + k
    
    # Load values
    # in_2[b, m, 0, k]
    val_in_2 = tl.load(in_2_ptr + in_2_offset)
    
    # in_1[0, 0, n, k] for both n=0 and n=1
    # For n=0:
    val_in_1_n0 = tl.load(in_1_ptr + 0 * K + k)  # in_1[..., 0, k]
    # For n=1:
    val_in_1_n1 = tl.load(in_1_ptr + 1 * K + k)  # in_1[..., 1, k]
    
    # Compute tmp_1 = in_2 * in_1 for both n
    tmp_1_n0 = val_in_2 * val_in_1_n0
    tmp_1_n1 = val_in_2 * val_in_1_n1
    
    # Load in_0 for both n
    # in_0[0, k]
    val_in_0_n0 = tl.load(in_0_ptr + 0 * K + k)
    # in_0[1, k]  
    val_in_0_n1 = tl.load(in_0_ptr + 1 * K + k)
    
    # Compute tmp_2 = tmp_1 + in_0 for both n
    tmp_2_n0 = tmp_1_n0 + val_in_0_n0
    tmp_2_n1 = tmp_1_n1 + val_in_0_n1
    
    # Now handle unbind and permute:
    # tmp_3[0] = tmp_2[..., 0, :] -> out_1 [b, m, k] (this is tmp_4)
    # tmp_3[1] = tmp_2[..., 1, :] -> tmp_5 -> permute -> out_0 [b, k, m] (this is tmp_6)
    
    # Store to out_1 (tmp_4)
    tl.store(out_1_ptr + out_1_offset, tmp_2_n0)
    
    # Store to out_0 (tmp_6 = permute(tmp_5))
    tl.store(out_0_ptr + out_0_offset, tmp_2_n1)


@torch.fx.wrap
def fused_mul_add_unbind_permute_wrapper(in_0, in_1, in_2):
    """
    Wrapper function for the fused kernel.
    
    Input shapes:
    - in_0: [2, K] = [2, 128]
    - in_1: [1, 1, 2, K] = [1, 1, 2, 128]
    - in_2: [B, M, 1, K] = [B, 17, 1, 128]
    
    Output:
    - out_0: [B, K, M] = [B, 128, 17] (tmp_6)
    - out_1: [B, M, K] = [B, 17, 128] (tmp_4)
    """
    B = in_2.shape[0]
    M = in_2.shape[1]
    K = in_2.shape[3]  # 128
    N = 2  # The unbind dimension
    
    # Reshape in_1 from [1, 1, 2, K] to [2, K] by squeezing
    # Since in_1 is [1, 1, 2, K], we can view it as [2, K] 
    # (squeeze the first two dimensions)
    in_1_squeezed = in_1.squeeze(0).squeeze(0)  # [2, K]
    
    # Output shapes
    out_0 = torch.empty((B, K, M), dtype=in_0.dtype, device=in_0.device)  # [B, 128, 17]
    out_1 = torch.empty((B, M, K), dtype=in_0.dtype, device=in_0.device)  # [B, 17, 128]
    
    # Calculate strides for in_2
    stride_in_2_m = in_2.stride(1)
    stride_in_2_k = in_2.stride(3)
    
    # BLOCK_SIZE for Triton
    BLOCK_SIZE = 128
    
    # Calculate grid
    # We parallelize over B * M * K elements
    num_elements = B * M * K
    grid = (num_elements,)
    
    fused_mul_add_unbind_permute_kernel_v2[grid](
        in_0, in_1_squeezed, in_2,
        out_0, out_1,
        B, M, N, K,
        in_0.stride(0),  # stride_in_0
        in_1_squeezed.stride(0), in_1_squeezed.stride(1),  # in_1 strides
        stride_in_2_m, stride_in_2_k,  # in_2 strides
        BLOCK_SIZE
    )
    
    return out_0, out_1


def pattern(in_0, in_1, in_2):
    """
    Pattern matching the computation:
    tmp_0 = in_0
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + tmp_0
    tmp_3 = torch.unbind(tmp_2, dim=2)
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = tmp_5.permute(0, 2, 1)
    return (tmp_6, tmp_4)
    """
    # Match exactly the operations in model.py
    # The pattern should match: mul + add + unbind + permute
    t0 = in_0          # tmp_0 = in_0
    t1 = in_2 * in_1   # tmp_1 = in_2 * in_1
    t2 = t1 + t0       # tmp_2 = tmp_1 + tmp_0
    t3 = torch.unbind(t2, dim=2)  # tmp_3 = torch.unbind(tmp_2, dim=2)
    t4 = t3[0]        # tmp_4 = tmp_3[0]
    t5 = t3[1]        # tmp_5 = tmp_3[1]
    t6 = t5.permute(0, 2, 1)  # tmp_6 = tmp_5.permute(0, 2, 1)
    # Return both values that are used in the final output
    return t6, t4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_mul_add_unbind_permute_wrapper