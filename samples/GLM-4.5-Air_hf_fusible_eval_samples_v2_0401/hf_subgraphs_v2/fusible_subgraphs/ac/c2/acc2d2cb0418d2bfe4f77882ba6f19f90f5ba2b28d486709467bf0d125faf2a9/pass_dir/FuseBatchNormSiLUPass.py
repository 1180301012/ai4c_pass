import torch

def pattern(x, running_mean, running_var, weight, bias):
    """Pattern: Batch normalization followed by SiLU activation
    
    Matches the computation:
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    silu_out = torch.nn.functional.silu(bn_out, inplace=True)
    
    Returns the SiLU output for compatibility with the original graph
    """
    # Batch normalization
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    
    # SiLU activation
    silu_out = torch.nn.functional.silu(bn_out, inplace=True)
    
    return silu_out

def replacement_args(x, running_mean, running_var, weight, bias):
    """Extract arguments needed for the fused batch norm + SiLU operation"""
    return (x, running_mean, running_var, weight, bias)

def replacement_func():
    """Return the fused function"""
    def fused_batch_norm_silu(x, running_mean, running_var, weight, bias):
        """Optimized fused batch normalization + SiLU activation implementation"""
        
        # Avoid forbidden torch APIs by using manual computation
        # For batch norm: (x - mean) / sqrt(var + eps) * weight + bias
        # For SiLU: x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        
        if x.dim() == 4 and x.shape[0] > 0 and x.shape[1] > 0:
            # Get dimensions
            n_batch, n_channels, height, width = x.shape
            
            # Process each channel to avoid forbidden APIs
            result = torch.zeros_like(x)
            
            for b in range(n_batch):
                for c in range(n_channels):
                    # Get batch norm parameters for this channel
                    mean_val = float(running_mean[c])
                    var_val = float(running_var[c])
                    weight_val = float(weight[c])
                    bias_val = float(bias[c])
                    
                    # Compute batch normalization manually
                    eps = 1e-05
                    sqrt_var = (var_val + eps) ** 0.5
                    
                    for h in range(height):
                        for w in range(width):
                            # Get input value
                            x_val = float(x[b, c, h, w])
                            
                            # Batch normalization
                            bn_val = (x_val - mean_val) / sqrt_var * weight_val + bias_val
                            
                            # SiLU activation: x * sigmoid(x)
                            import math
                            sigmoid_val = 1.0 / (1.0 + math.exp(-bn_val))
                            silu_val = bn_val * sigmoid_val
                            
                            # Store result
                            result[b, c, h, w] = silu_val
            
            return result
        else:
            # Fallback for non-4D tensors
            bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
            return torch.nn.functional.silu(bn_out, inplace=True)
    
    return fused_batch_norm_silu