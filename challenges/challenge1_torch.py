import torch
import xlns as xl

# Create an xlns object
xlns_data = xl.xlnsnp([2.0, 4.0, 5.0])
print("xlns data:", xlns_data)

# Try to create a PyTorch tensor directly from an xlns object
try:
    pt_data = torch.tensor(xlns_data)
except AttributeError as e:
    # Display only the first line of the error message
    error_str = str(e).splitlines()[0]
    print("\nPyTorch error when passing xlns object:", error_str)

# The correct approach is to convert to floating point first:
fp_data = [float(x) for x in xlns_data.xlns()]
pt_data = torch.tensor(fp_data)
print("\nConverted PyTorch tensor:", pt_data)