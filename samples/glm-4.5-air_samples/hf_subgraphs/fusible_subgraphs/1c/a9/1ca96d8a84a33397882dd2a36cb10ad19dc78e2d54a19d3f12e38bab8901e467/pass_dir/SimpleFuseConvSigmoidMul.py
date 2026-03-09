import torch
import torch._refs


def pattern(in_6, in_1, in_0, in_5):
    """
    Pattern: conv2d -> sigmoid -> mul
    """
    tmp_2 = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.sigmoid(tmp_2)
    tmp_4 = in_5 * tmp_3
    return tmp_2, tmp_3, tmp_4


def replacement_args(in_6, in_1, in_0, in_5):
    return (in_6, in_1, in_0, in_5)


def replacement_func():
    def replacement(in_6, in_1, in_0, in_5):
        # Use 1x1 conv which is a simple linear transform
        # For 1x1 conv with [B, 10, 1, 1] input and [40, 10, 1, 1] weight
        # This is equivalent to: output[b, m] = sum_k input[b, k] * weight[m, k] + bias[m]
        
        # Reshape for efficient computation
        B = in_6.shape[0]
        K = in_6.shape[1]   # 10
        M = in_1.shape[0]   # 40
        H = in_5.shape[2]
        W = in_5.shape[3]
        
        # Flatten
        in_6_flat = in_6.view(B, K)  # [B, K]
        in_1_flat = in_1.view(M, K)  # [M, K]
        
        # Matrix multiply: [B, M] = [B, K] @ [M, K].T
        tmp_2 = torch.matmul(in_6_flat, in_1_flat.t()) + in_0  # [B, M]
        tmp_2 = tmp_2.view(B, M, 1, 1)  # [B, M, 1, 1]
        
        # Sigmoid
        tmp_3 = torch.sigmoid(tmp_2)  # [B, M, 1, 1]
        
        # Multiply with broadcast
        tmp_4 = tmp_3 * in_5  # [B, M, H, W]
        
        return tmp_2, tmp_3, tmp_4
    
    return replacement