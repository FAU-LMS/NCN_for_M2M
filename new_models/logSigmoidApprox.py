import torch


class logSigmoidApprox(torch.nn.Module):
    def __init__(self):
        super(logSigmoidApprox, self).__init__()

    def forward(self, vals):
        x, y = torch.load("new_models/logsigmoid_params.pth")
        results = torch.zeros_like(vals)
        results[vals < x[0]] = vals[vals < x[0]]
        for i in range(len(x)-1):
            mask = torch.logical_and(vals >= x[i], vals < x[i+1])
            results[mask] = y[i] + (vals[mask] - x[i]) * (y[i+1] - y[i])/(x[i+1] - x[i])
        return results
