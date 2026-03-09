import torch

def pattern(in_0, in_1, in_2, in_3):
    # Simple pattern - just match two contiguous operations
    contig_1 = in_1.contiguous()
    contig_2 = in_2.contiguous()
    return (contig_1, contig_2)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

def replacement_func():
    def optimize_contiguous(in_0, in_1, in_2, in_3):
        # Simple pattern - just optimize two contiguous operations
        contig_1 = in_1.contiguous()
        contig_2 = in_2.contiguous()
        return (contig_1, contig_2)
    
    return optimize_contiguous