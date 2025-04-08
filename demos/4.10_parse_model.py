"""
See section 4.10: Benchmarking and Visualisation Tools
Shows how we can parse a model and get all its children
(layers) in a flat list. This can be used when we want
to benchmark and compare the performance of an entire
model with different precisions or with standard FP.
"""
import torch

class TestModel(torch.nn.Module):

    def __init__(self):
        super(TestModel, self).__init__()
        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x

def get_children(model):
    children = list(model.children())
    flat_children = []
    if not children:
        return model
    for child in children:
        # get_children returns either a torch.nn.Module or a list
        try:
            flat_children.extend(get_children(child))
        except TypeError:
            flat_children.append(get_children(child))
    return flat_children

print(*get_children(TestModel()), sep="\n")

# Output:
# Linear(in_features=100, out_features=200, bias=True)
# ReLU()
# Linear(in_features=200, out_features=10, bias=True)
# Softmax(dim=None)