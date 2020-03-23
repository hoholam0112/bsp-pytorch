from torch.autograd import Function

class GradientReversalLayer(Function):
    """ Negate gradient in backward pass """

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None
