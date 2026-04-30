import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 256}, num_warps=8),
    ],
    key=["K"],
)
@triton.jit
def fused_gated_update_kernel(
    # Input pointers
    in_8_ptr, in_9_ptr, in_10_ptr, in_11_ptr,
    # Weight pointers for linear
    in_7_ptr, in_6_ptr,
    # LayerNorm params (linear output)
    in_3_ptr, in_2_ptr,
    # LayerNorm params (in_10)
    in_1_ptr, in_0_ptr,
    # LayerNorm params (in_11)
    in_5_ptr, in_4_ptr,
    # Output pointer
    out_ptr,
    # Shapes
    M, K,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for gated update computation.
    """
    pid = tl.program_id(0)
    
    # Each program handles one row in M dimension
    row_offset = pid * K
    
    # Cast to float32 for computation (required for exp and sqrt precision)
    DTYPE_FP32 = tl.float32
    
    # Compute linear: in_8[m, 0, :] @ in_7[:, :].t() + bias
    # in_8[m, 0, :] is [1, K], in_7 is [K, K]
    # Linear output is [1, K] = [1, K] @ [K, K].t() = [1, K] @ [K, K]
    # So we need in_8[i] * in_7[j, i] for each output j
    linear_vals = tl.zeros((BLOCK_SIZE,), dtype=DTYPE_FP32)
    
    # Load in_8[m, 0, :] row
    in_8_row = tl.load(
        in_8_ptr + row_offset + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < K,
        other=0.0
    ).to(DTYPE_FP32)
    
    # Compute matrix-vector product: for each output element j
    for j in range(0, K, BLOCK_SIZE):
        mask_j = j + tl.arange(0, BLOCK_SIZE) < K
        
        # Load in_7[:, j] - column j of in_7 (j-th output feature weights)
        # in_7 is [K, K] stored row-major, so column j is at offset j, j+K, j+2K, ...
        in_7_col = tl.load(
            in_7_ptr + j + tl.arange(0, BLOCK_SIZE) * K,
            mask=mask_j,
            other=0.0
        ).to(DTYPE_FP32)
        
        # Dot product: sum_i in_8_row[i] * in_7_col[i]
        # in_7_col[i] = in_7[i, j], so sum_i in_8_row[i] * in_7[i, j] is correct
        linear_vals = linear_vals + tl.sum(in_8_row * in_7_col)
    
    # Add bias
    linear_vals = linear_vals + tl.load(in_6_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32)
    
    # LayerNorm on linear output
    mean = tl.sum(linear_vals, axis=0) / K
    var = tl.sum((linear_vals - mean) * (linear_vals - mean), axis=0) / K
    rstd = 1.0 / tl.sqrt(var + eps)
    ln1 = (linear_vals - mean) * rstd
    ln1 = ln1 * tl.load(in_3_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32) + tl.load(in_2_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32)
    
    # tmp_11 = tmp_9.sigmoid() = sigmoid(layer_norm(linear_output))
    sig1 = 1.0 / (1.0 + tl.exp(-ln1))
    
    # Sigmoid on in_9[m, 0, :]
    in_9_vals = tl.load(
        in_9_ptr + row_offset + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < K,
        other=0.0
    ).to(DTYPE_FP32)
    sig2 = 1.0 / (1.0 + tl.exp(-in_9_vals))
    
    # LayerNorm on in_11[m, :] - note: this is [M, K] with shape [300, 256]
    in_11_vals = tl.load(
        in_11_ptr + pid * K + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < K,
        other=0.0
    ).to(DTYPE_FP32)
    mean_ln = tl.sum(in_11_vals, axis=0) / K
    var_ln = tl.sum((in_11_vals - mean_ln) * (in_11_vals - mean_ln), axis=0) / K
    rstd_ln = 1.0 / tl.sqrt(var_ln + eps)
    ln2 = (in_11_vals - mean_ln) * rstd_ln
    ln2 = ln2 * tl.load(in_5_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32) + tl.load(in_4_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32)
    
    # Branch 1: sig1 * ln2 (after unsqueeze(-2), but we handle broadcasting implicitly)
    branch1 = sig1 * ln2
    
    # LayerNorm on in_10[m, 0, :]
    in_10_vals = tl.load(
        in_10_ptr + row_offset + tl.arange(0, BLOCK_SIZE),
        mask=tl.arange(0, BLOCK_SIZE) < K,
        other=0.0
    ).to(DTYPE_FP32)
    mean_ln3 = tl.sum(in_10_vals, axis=0) / K
    var_ln3 = tl.sum((in_10_vals - mean_ln3) * (in_10_vals - mean_ln3), axis=0) / K
    rstd_ln3 = 1.0 / tl.sqrt(var_ln3 + eps)
    ln3 = (in_10_vals - mean_ln3) * rstd_ln3
    ln3 = ln3 * tl.load(in_1_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32) + tl.load(in_0_ptr + tl.arange(0, BLOCK_SIZE)).to(DTYPE_FP32)
    
    # Branch 2: sig2 * ln3
    branch2 = sig2 * ln3
    
    # Final output: branch1 + branch2
    out_val = branch1 + branch2
    
    # Store result
    tl.store(out_ptr + row_offset + tl.arange(0, BLOCK_SIZE), out_val, mask=tl.arange(0, BLOCK_SIZE) < K)


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    """
    Match the full gated update computation:
    - linear(in_8, in_7, in_6)
    - layer_norm(linear, ...) -> sigmoid (tmp_11)
    - sigmoid(in_9) (tmp_10)
    - layer_norm(in_11, ...).unsqueeze * sigmoid(ln)
    - layer_norm(in_10, ...) * sigmoid(in_9)
    - Sum of both branches
    """
    linear_out = torch.nn.functional.linear(in_8, in_7, in_6)
    tmp_9 = torch.nn.functional.layer_norm(linear_out, (256,), in_3, in_2, 1e-05)
    
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()  # sigmoid on layer_norm output
    
    tmp_12 = torch.nn.functional.layer_norm(in_11, (256,), in_5, in_4, 1e-05)
    tmp_13 = torch.nn.functional.layer_norm(in_10, (256,), in_1, in_0, 1e-05)
    
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    
    return tmp_17


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11)


@torch.fx.wrap
def fused_gated_update_wrapper(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7, in_8, in_9, in_10, in_11):
    """
    Wrapper for the fused gated update kernel.
    Handles the kernel launch and memory management.
    """
    M = in_8.shape[0]  # 300
    N = in_8.shape[1]  # 1
    K = in_8.shape[2]  # 256
    
    num_programs = M  # One program per batch element
    output = torch.empty((M, N, K), dtype=in_8.dtype, device=in_8.device)
    
    fused_gated_update_kernel[(num_programs,)](
        in_8, in_9, in_10, in_11,
        in_7, in_6,
        in_3, in_2,
        in_1, in_0,
        in_5, in_4,
        output,
        M, K,
        1e-05,
    )
    
    return output


def replacement_func():
    return fused_gated_update_wrapper