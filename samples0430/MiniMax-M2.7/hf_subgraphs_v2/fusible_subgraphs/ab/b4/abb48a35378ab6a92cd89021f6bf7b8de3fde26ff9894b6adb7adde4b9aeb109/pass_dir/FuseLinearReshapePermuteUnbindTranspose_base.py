import torch
import triton
import triton.language as tl


@triton.jit
def fused_qkv_kernel_base(
    # Input pointers
    x_ptr,         # (B, N, Cin)
    weight_ptr,    # (Cout, Cin) - [2304, 768]
    # Output pointers
    q_out_ptr,     # (B, H, N, D) - [1, 16, 197, 48]
    k_out_ptr,     # (B, H, D, N) - [1, 16, 48, 197]
    v_out_ptr,     # (B, H, N, D) - [1, 16, 197, 48]
    # Shape info
    B: tl.constexpr,
    N: tl.constexpr,
    Cin: tl.constexpr,
    Cout: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    # Tile sizes
    BLOCK_SIZE: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """
    Fused kernel for: linear + reshape + permute + unbind + transpose
    Convit_base version: H=16, D=48
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    
    n_offsets = tl.arange(0, TILE_N)
    d_offsets = tl.arange(0, BLOCK_SIZE)
    
    for n_start in range(0, N, TILE_N):
        n_offsets = n_start + tl.arange(0, TILE_N)
        n_mask = n_offsets < N
        
        acc_q = tl.zeros((TILE_N, BLOCK_SIZE), dtype=tl.float32)
        acc_k = tl.zeros((TILE_N, BLOCK_SIZE), dtype=tl.float32)
        acc_v = tl.zeros((TILE_N, BLOCK_SIZE), dtype=tl.float32)
        
        for cin_start in range(0, Cin, BLOCK_SIZE):
            cin_offsets = cin_start + tl.arange(0, BLOCK_SIZE)
            cin_mask = cin_offsets < Cin
            
            x_ptrs = x_ptr + pid_b * N * Cin + n_offsets[:, None] * Cin + cin_offsets[None, :]
            x = tl.load(x_ptrs, mask=cin_mask[None, :] & n_mask[:, None], other=0.0)
            
            for sub_head in range(3):
                w_row_offset = (sub_head * H + pid_h) * D
                w_ptrs = weight_ptr + (w_row_offset + d_offsets[None, :]) * Cin + cin_offsets[:, None]
                w = tl.load(w_ptrs, mask=cin_mask[:, None] & (d_offsets[None, :] < D), other=0.0)
                
                if sub_head == 0:
                    acc_q += tl.dot(x, w)
                elif sub_head == 1:
                    acc_k += tl.dot(x, w)
                else:
                    acc_v += tl.dot(x, w)
        
        scale = 1.0 / tl.sqrt(tl.cast(D, tl.float32))
        acc_q = acc_q * scale
        acc_k = acc_k * scale
        acc_v = acc_v * scale
        
        q_offset = ((pid_b * H + pid_h) * N + n_offsets[:, None]) * D + d_offsets[None, :]
        tl.store(q_out_ptr + q_offset, acc_q, mask=n_mask[:, None] & (d_offsets[None, :] < D))
        
        k_offset = ((pid_b * H + pid_h) * D + d_offsets[:, None]) * N + n_offsets[None, :]
        tl.store(k_out_ptr + k_offset, acc_k, mask=(d_offsets[:, None] < D) & n_mask[None, :])
        
        v_offset = ((pid_b * H + pid_h) * N + n_offsets[:, None]) * D + d_offsets[None, :]
        tl.store(v_out_ptr + v_offset, acc_v, mask=n_mask[:, None] & (d_offsets[None, :] < D))


@torch.fx.wrap
def fused_qkv_wrapper_base(x, weight, B, N, Cin, Cout, H, D):
    """
    Wrapper for convit_base: H=16, D=48
    """
    Q = torch.empty((B, H, N, D), device=x.device, dtype=x.dtype)
    K_T = torch.empty((B, H, D, N), device=x.device, dtype=x.dtype)
    V = torch.empty((B, H, N, D), device=x.device, dtype=x.dtype)
    
    TILE_N = min(64, N)
    BLOCK_SIZE = min(64, D)
    
    grid = (B, H)
    
    fused_qkv_kernel_base[grid](
        x, weight, Q, K_T, V,
        B, N, Cin, Cout, H, D,
        BLOCK_SIZE=BLOCK_SIZE,
        TILE_N=TILE_N,
    )
    
    return Q, K_T, V


def pattern(in_0, in_1):
    """
    Match the pattern: linear + reshape + permute + unbind + transpose
    Convit_base version: reshape(1, 197, 3, 16, 48), H=16, D=48
    """
    tmp_1 = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = tmp_1.reshape(1, 197, 3, 16, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    
    return tmp_5, tmp_8, tmp_7


def replacement_args(in_0, in_1):
    """Extract arguments needed for the replacement function."""
    return (in_0, in_1)


def replacement_func():
    return fused_qkv_dispatcher_base


@torch.fx.wrap
def fused_qkv_dispatcher_base(in_0, in_1):
    """
    Dispatcher for convit_base.
    Shape: [2304, 768] x [1, 197, 768] -> [1, 197, 2304] -> reshape -> permute -> unbind
    Output: Q=(1,16,197,48), K^T=(1,16,48,197), V=(1,16,197,48)
    """
    weight = in_0  # [2304, 768]
    x = in_1       # [1, 197, 768]
    
    B, N, Cin = x.shape  # 1, 197, 768
    Cout = weight.shape[0]  # 2304
    H = 16
    D = 48
    
    return fused_qkv_wrapper_base(x, weight, B, N, Cin, Cout, H, D)