"""Six-layer BP network with the original matmul, bias, ReLU, and BN order."""
import numpy as np
import torch
from torch import nn


class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, weight, bias):
        ctx.save_for_backward(inputs, weight, bias)
        output = inputs.matmul(weight.t())
        output += bias.unsqueeze(0).expand_as(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, weight, bias = ctx.saved_tensors
        grad_input = grad_output.matmul(weight) if ctx.needs_input_grad[0] else None
        grad_weight = grad_output.t().matmul(inputs) if ctx.needs_input_grad[1] else None
        grad_bias = grad_output.sum(0).squeeze(0) if ctx.needs_input_grad[2] else None
        return grad_input, grad_weight, grad_bias


class Linear(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(np.random.normal(0,np.sqrt(2/inputs),(outputs,inputs)),dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(outputs))

    def forward(self, inputs):
        return LinearFunction.apply(inputs,self.weight,self.bias)


class LinearNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        widths = [3072]+[256]*5+[100]
        self.linear = nn.ModuleList([Linear(a,b) for a,b in zip(widths,widths[1:])])
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(b) for b in widths[1:]])
        self.relu = nn.ModuleList([nn.ReLU() for _ in widths[1:]])

    def forward(self, images):
        value = images.view(-1,3072)
        for i in range(5):
            value = self.batchnorm[i](self.relu[i](self.linear[i](value)))
        return self.linear[-1](value)
