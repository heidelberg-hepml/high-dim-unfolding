import torch
import torch.distributions as D

class GammaMixture(D.MixtureSameFamily):
    def __init__(self, params):
        assert len(params.shape) == 3 and params.shape[-1] == 3
        self.params = params
        mix = D.Categorical(torch.ones_like(params[:, :, 2]))
        gammas = D.Gamma(params[:, :, 0], params[:, :, 1])
        super().__init__(mix, gammas)

    def cross_entropy(self, target):
        assert target.shape == self.batch_shape
        return -self.log_prob(target)

    def smooth_cross_entropy(self, target, max_num_particles, spread):
        assert target.shape == self.batch_shape
        logprobs = self.log_prob(
            torch.arange(1, max_num_particles + 1, device=target.device)
            .unsqueeze(1)
            .repeat(1, len(target))
        ).transpose(0, 1)
        weights = torch.zeros_like(logprobs, device=target.device)
        for i in range(len(target)):
            weights[i, target[i] - 1] = 1.0
        kernel = (
            torch.distributions.Normal(0, spread)
            .log_prob(torch.arange(-2, 3, device=target.device))
            .exp()
        )
        kernel /= kernel.sum()
        weights = torch.nn.functional.conv1d(
            weights.unsqueeze(1), kernel.view(1, 1, -1), bias=None, groups=1, padding=2
        ).squeeze(1)
        return torch.sum(-logprobs * weights, dim=-1)
