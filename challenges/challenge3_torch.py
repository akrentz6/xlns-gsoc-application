import torch
import xlns as xl

# Constants
xlnsB_pt = torch.tensor(xl.xlnsB, dtype=torch.float64)
xlns0_pt = torch.tensor(-0x8000000000000000, dtype=torch.int64)

def xlns_to_torch(xlns_array):
    return torch.tensor(xlns_array.nd, dtype=torch.int64)

def torch_to_xlns(torch_tensor):
    
    internal_data = torch_tensor.cpu().numpy()
    fp_values = []
    
    # Set the internal data of each xlns object
    for value in internal_data:
        x_obj = xl.xlns(0)
        x_obj.x = value >> 1
        x_obj.s = value & 1
        fp_values.append(x_obj)
    
    return xl.xlnsnp(fp_values)

# Logarithm term of addition equation
def torch_sbdb_ideal(d, s, B=None):

    d_f = d.to(torch.float64)
    s_f = s.to(torch.float64)

    base = xlnsB_pt if B is None else torch.tensor(B, dtype=torch.float64, device=d.device)
    term = torch.pow(base, d_f)

    v = torch.abs(1.0 - 2.0 * s_f + term)
    log_val = torch.log(v) / torch.log(base)

    result = torch.bitwise_left_shift(torch.round(log_val).to(torch.int64), 1)
    return result

def torch_logadd(x, y):

    # Our formulae suppose the log part of x is greater than that of y.
    # We are able to take the maximum of the full internal representation
    # since the sign bit is the least significant bit so doesn't matter. 
    part1 = torch.maximum(x, y)

    d = -torch.abs(torch.bitwise_right_shift(x, 1) - torch.bitwise_right_shift(y, 1))
    s = torch.bitwise_and(torch.bitwise_xor(x, y), 1)
    part2 = torch_sbdb_ideal(d, s)

    return part1 + part2

def torch_myadd(x, y):

    # Handles logic for when their sum equals 0. In this case, the 
    # internal representations differ only in their LSB, i.e. x^y=1.
    sum_to_zero_cond = torch.where(torch.eq(torch.bitwise_xor(x, y), 1),
                                   xlns0_pt, torch_logadd(x, y))
    
    # Handles logic for when either term equals 0.
    y_equals_zero_cond = torch.where(torch.eq(torch.bitwise_or(y, 1),
                                              xlns0_pt),
                                     x, sum_to_zero_cond)
    x_equals_zero_cond = torch.where(torch.eq(torch.bitwise_or(x, 1),
                                              xlns0_pt),
                                     y, y_equals_zero_cond)
    
    return x_equals_zero_cond

if __name__ == '__main__':

    # Test data including edge cases (negatives and numbers that sum to 0)
    # x = torch.Tensor([16777216 26591258 -9223372036854775808 0])
    # y = torch.Tensor([0 1 16777216 1])
    x = xlns_to_torch(xl.xlnsnp([2,  3,  0,  1]))
    y = xlns_to_torch(xl.xlnsnp([1, -1,  2, -1]))
    
    result = torch_myadd(x, y)
    print("PyTorch myadd result:", torch_to_xlns(result))