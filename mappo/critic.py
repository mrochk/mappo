import torch as th
from torch import nn
from torch.nn import functional as F

class Critic(nn.Module):
    def __init__(
        self, 
        state_space_size: int, 
        hidden_size: int = 64,
        batch_size: int = 64,
        epochs: int = 10
    ):

        super().__init__()

        self.state_space_size = state_space_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.fc1 = nn.Linear(state_space_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.optim = th.optim.Adam(self.parameters(), lr=0.0003)

    def forward(self, state: th.Tensor) -> th.Tensor:
        z1 = F.relu(self.fc1(state))
        z2 = F.relu(self.fc2(z1))
        z3 = self.fc3(z2)
        return z3

    def value(self, state: th.Tensor):
        state = state.flatten(start_dim=-2)
        return self.forward(state)

    def update(self, dataset: dict):
        for epoch in range(self.epochs):
            loss = th.zeros(1, dtype=th.float32, requires_grad=True)

            for agent in dataset.keys():
                data = dataset[agent]

                length = data['states'].shape[0]
                batch_indices = th.randperm(length)[self.batch_size]

                states = data['states'][batch_indices]
                returns = data['returns'][batch_indices]

                predictions = self.value(states)
                loss = loss + F.mse_loss(predictions, returns)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def compute_gae(self, rewards: th.Tensor, values: th.Tensor, gamma: float, lam: float):
        # compute deltas
        deltas = rewards + gamma * values[1:] - values[:-1]

        # compute advantages: discounter cumsum of deltas
        advantages, gae = [], 0.0
        for delta in deltas:
            gae = delta + gamma * lam * gae
            advantages.append(gae)

        advantages = values.new_tensor(list(reversed(advantages)))

        # compute returns
        returns = advantages + values[:-1]
        return advantages, returns

    def add_gae(self, trajectories: dict, gamma: float = 0.99, lam: float = 0.95):
        # collect all advantages to normalize across batch
        all_advs = []

        # first pass: compute un-normalized advantages & returns
        for trajs in trajectories.values():
            for trajectory in trajs:
                rewards = trajectory['rewards']
                # get values for all states plus the terminal next‐state
                with th.no_grad():
                    values = th.cat([self.state_value(trajectory['states']), th.zeros(1)])

                advantages, returns = self.compute_gae(rewards, values, gamma, lam)

                trajectory['advantages'] = advantages
                trajectory['rtgs']       = returns
                all_advs.append(advantages)

        # flatten and normalize advantages over entire batch
        flat_advs = th.cat(all_advs)
        adv_mean, adv_std = flat_advs.mean(), flat_advs.std(unbiased=False) + 1e-8

        # second pass: write back normalized advantages
        for trajs in trajectories.values():
            for trajectory in trajs:
                trajectory['advantages'] = (trajectory['advantages'] - adv_mean) / adv_std