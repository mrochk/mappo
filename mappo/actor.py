import torch as th
from torch import nn
from torch.distributions import Categorical

class Actor(nn.Module):
    '''PPO actor network module.'''
    def __init__(self, state_space_size: int, n_actions: int):
        super().__init__()

        self.state_space_size = state_space_size

        self.layers = nn.Sequential(
            nn.Linear(state_space_size, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

        self.optim = th.optim.Adam(self.parameters(), 0.0003)

    def distribution(self, state) -> Categorical:
        state = th.as_tensor(state, dtype=th.float32)
        state = th.flatten(state, start_dim=-2)
        logits = self.layers(state)
        actions_dist = Categorical(logits=logits)
        return actions_dist

    def action(self, state) -> tuple[float, float]:
        actions_dist = self.distribution(state)
        action  = actions_dist.sample()
        logprob = actions_dist.log_prob(action)
        return action.item(), logprob

    def loss(self, dataset: dict, indices, eps: float, c_entropy: float) -> th.Tensor:
        states       = dataset['states'][indices]
        actions      = dataset['actions'][indices]
        logprobs_old = dataset['logprobs'][indices]
        advantages   = dataset['advantages'][indices]

        distributions = self.distribution(states)
        entropy = distributions.entropy().mean()
        loss_entropy = c_entropy * entropy

        logprobs_new = distributions.log_prob(actions)
        ratio = th.exp(logprobs_new - logprobs_old)
        clipped = th.clip(ratio, 1.0 - eps, 1.0 + eps) * advantages
        loss_clip = th.min(ratio * advantages, clipped).mean()

        return loss_clip + loss_entropy

    def update(self, dataset: th.Tensor, epochs: int, eps: float, c_entropy: float, batch_size: int):
        for epoch in range(epochs):
            batch_indices = th.randperm(dataset['rewards'].shape[0])[:batch_size]
            loss = -self.loss(dataset, batch_indices, eps, c_entropy)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
