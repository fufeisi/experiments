import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

def unsqueeze_all(t):
    # Helper function to unsqueeze all the dimensions that we reduce over
    return t[None, :, None, None]

def batch_norm_backward(grad_out, X, sum, sqrt_var, N, eps):
    # We use the formula: out = (X - mean(X)) / (sqrt(var(X)) + eps)
    # in batch norm 2d's forward. To simplify our derivation, we follow the
    # chain rule and compute the gradients as follows before accumulating
    # them all into a final grad_input.
    #  1) 'grad of out wrt var(X)' * 'grad of var(X) wrt X'
    #  2) 'grad of out wrt mean(X)' * 'grad of mean(X) wrt X'
    #  3) 'grad of out wrt X in the numerator' * 'grad of X wrt X'
    # We then rewrite the formulas to use as few extra buffers as possible
    tmp = ((X - unsqueeze_all(sum) / N) * grad_out).sum(dim=(0, 2, 3))
    tmp *= -1
    d_denom = tmp / (sqrt_var + eps)**2  # d_denom = -num / denom**2
    # It is useful to delete tensors when you no longer need them with `del`
    # For example, we could've done `del tmp` here because we won't use it later
    # In this case, it's not a big difference because tmp only has size of (C,)
    # The important thing is avoid allocating NCHW-sized tensors unnecessarily
    d_var = d_denom / (2 * sqrt_var)  # denom = torch.sqrt(var) + eps
    # Compute d_mean_dx before allocating the final NCHW-sized grad_input buffer
    d_mean_dx = grad_out / unsqueeze_all(sqrt_var + eps)
    d_mean_dx = unsqueeze_all(-d_mean_dx.sum(dim=(0, 2, 3)) / N)
    # d_mean_dx has already been reassigned to a C-sized buffer so no need to worry

    # (1) unbiased_var(x) = ((X - unsqueeze_all(mean))**2).sum(dim=(0, 2, 3)) / (N - 1)
    grad_input = X * unsqueeze_all(d_var * N)
    grad_input += unsqueeze_all(-d_var * sum)
    grad_input *= 2 / ((N - 1) * N)
    # (2) mean (see above)
    grad_input += d_mean_dx
    # (3) Add 'grad_out / <factor>' without allocating an extra buffer
    grad_input *= unsqueeze_all(sqrt_var + eps)
    grad_input += grad_out
    grad_input /= unsqueeze_all(sqrt_var + eps)  # sqrt_var + eps > 0!
    return grad_input

class BatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, eps=1e-3):
        # Don't save keepdim'd values for backward
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.save_for_backward(X)
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        X, = ctx.saved_tensors
        return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)

class qBatchNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, eps=1e-3):
        # Don't save keepdim'd values for backward
        sum = X.sum(dim=(0, 2, 3))
        var = X.var(unbiased=True, dim=(0, 2, 3))
        N = X.numel() / X.size(1)
        sqrt_var = torch.sqrt(var)
        ctx.save_for_backward(torch.quantize_per_tensor_dynamic(X, torch.quint8, False))
        ctx.eps = eps
        ctx.sum = sum
        ctx.N = N
        ctx.sqrt_var = sqrt_var
        mean = sum / N
        denom = sqrt_var + eps
        out = X - unsqueeze_all(mean)
        out /= unsqueeze_all(denom)
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_out):
        qX, = ctx.saved_tensors
        X = qX.dequantize()
        return batch_norm_backward(grad_out, X, ctx.sum, ctx.sqrt_var, ctx.N, ctx.eps)

class qBatchNormLayer(torch.nn.BatchNorm2d):
     def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        return qBatchNorm.apply(input)



if __name__ == '__main__':
     a = torch.rand(10, 20, 30, 4, requires_grad=True, dtype=torch.double)
     res = []
     res.append(torch.autograd.gradcheck(BatchNorm.apply, (a,), fast_mode=False))
     # res.append(torch.autograd.gradcheck(qBatchNorm.apply, (a,), fast_mode=False))
     print(res)