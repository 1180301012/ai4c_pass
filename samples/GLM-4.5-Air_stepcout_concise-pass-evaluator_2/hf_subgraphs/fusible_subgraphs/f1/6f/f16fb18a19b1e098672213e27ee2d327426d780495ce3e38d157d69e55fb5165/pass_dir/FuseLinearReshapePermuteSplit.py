import torch
import triton
import triton.language as tl

def pattern(in_1, weight):
    tmp_1 = torch.nn.functional.linear(in_1, weight, None)
    # The reshape dimension varies between models (9 for small, 4 for tiny)
    # We can infer this from the input shape
    tmp_2 = tmp_1.reshape(1, 197, 3, tmp_1.shape[2] // (3 * 48), 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    tmp_4 = tmp_3.unbind(0)
    tmp_5 = tmp_4[0]
    tmp_6 = tmp_4[1]
    tmp_7 = tmp_4[2]
    return tmp_5, tmp_6, tmp_7

def replacement_args(in_1, weight):
    return (in_1, weight)

@triton.jit
def fused_qkv_kernel(
    in_ptr,      # [B, N, C] input tensor 
    weight_ptr,  # [3*C_out, C_in] weight tensor
    q_ptr,       # [B, N, C_out] 
    k_ptr,       # [B, N, C_out]
    v_ptr,       # [B, N, C_out]
    b, n, c_in, c_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one head (Q, K, or V)
    head_id = tl.program_id(0)
    row_idx = tl.program_id(1)
    col_idx = tl.arange(0, BLOCK_SIZE)
    
    mask = col_idx < c_out
    
    # Calculate offset for this head (Q=0, K=1, V=2)
    head_offset = head_id * c_out
    
    # Load input row [1, C_in]
    in_offset = row_idx * c_in + col_idx
    x = tl.load(in_ptr + in_offset, mask=mask, other=0.0)
    
    # Load weight row [C_out, C_in] for this head
    # For Q head: weight[0:c_out, :], K head: weight[c_out:2*c_out, :], V head: weight[2*c_out:3*c_out, :]
    weight_offset = (head_offset + col_idx) * c_in
    w = tl.load(weight_ptr + weight_offset, mask=mask, other=0.0)
    
    # GEMM: x @ w^T
    acc = tl.sum(x * w, axis=0)
    
    # Store result for this head
    out_offset = (row_idx * c_out + col_idx) + head_offset * b * n * c_out
    tl.store(q_ptr if head_id == 0 else (k_ptr if head_id == 1 else v_ptr) + out_offset, acc, mask=mask)

@torch.fx.wrap
def fused_linear_qkv(in_1, weight):
    b, n, c_in = in_1.shape
    total_c_out = weight.shape[0]
    c_out = total_c_out // 3  # 3 heads (Q, K, V)
    
    # Allocate output tensors
    q = torch.empty((b, n, c_out), dtype=in_1.dtype, device=in_1.device)
    k = torch.empty((b, n, c_out), dtype=in_1.dtype, device=in_1.device)  
    v = torch.empty((b, n, c_out), dtype=in_1.dtype, device=in_1.device)
    
    # Set up kernel launch
    BLOCK_SIZE = 256
    num_programs = (c_out + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_qkv_kernel[(3, b, num_programs)](
        in_ptr=in_1,
        weight_ptr=weight,
        q_ptr=q,
        k_ptr=k, 
        v_ptr=v,
        b=b, n=n, c_in=c_in, c_out=c_out,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return q, k, v

def replacement_func():
    return fused_linear_qkv