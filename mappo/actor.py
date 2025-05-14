import numpy as np
import torch as th
from torch import Tensor
from torch import nn
from torch.distributions import Categorical

class Actor(nn.Module):
    '''PPO Actor Module.'''

    def __init__(self, statesize: int, n_actions: int, epochs: int, 
            batch_size: int = 64, eps: float = 0.2, c_ent: float = 0.0, lr: float = 0.0003):
        '''
        Args:
            statesize (int): State space size (size of first layer input).
            n_actions (int): Number of discrete actions the agent should output. 
            epochs (int): How many times we update when training.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            eps (float, optional): Epsilon in PPO^CLIP objective. Defaults to 0.2.
            c_ent (float, optional): Entropy coefficient. Defaults to 0.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.0003.
        '''

        super().__init__()
        self.ssize, self.epochs, self.batch_size, self.eps, self.c_ent = statesize, epochs, batch_size, eps, c_ent

        self.policy_net = nn.Sequential(
            nn.Linear(statesize, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
        )

        self.action_net = nn.Linear(512, n_actions)

        # the separation in policy_net and action_net is made to copy the sb3 implementation
        # such that we can then load pretrained weights from agents trained with sb3 

        self.optim = th.optim.Adam(self.parameters(), lr)

    def forward(self, state: np.array) -> Tensor:
        '''Compute logits for a given game state.'''
        if not isinstance(state, Tensor): state = th.as_tensor(state, dtype=th.float32)
        if not state.shape[-1] == self.ssize: state = th.flatten(state, start_dim=-2)
        logits = self.action_net(self.policy_net(state))
        return logits

    def distribution(self, state: np.array) -> Categorical:
        '''Outputs a probability distribution over discrete actions for a given game state'''
        logits = self.forward(state)
        actions_dist = Categorical(logits=logits)
        return actions_dist

    def action(self, state: np.array) -> tuple[int, th.Tensor]:
        '''Output a pair of (action, logprob) for a given game state.'''
        actions_dist = self.distribution(state)
        action  = actions_dist.sample()
        logprob = actions_dist.log_prob(action)
        return action.item(), logprob

    def objective(self, dataset: dict, indices) -> th.Tensor:
        '''PPO Objective Function'''

        # extract relevant elements
        states = dataset['states'][indices]
        actions = dataset['actions'][indices]
        logprobs_old = dataset['logprobs'][indices]
        advantages = dataset['advantages'][indices]

        # get distributions
        distributions = self.distribution(states)

        # entropy
        entropy = distributions.entropy().mean()
        loss_entropy = self.c_ent * entropy

        # clipped loss
        logprobs_new = distributions.log_prob(actions)
        ratio = th.exp(logprobs_new - logprobs_old)
        clipped = th.clip(ratio, 1.0 - self.eps, 1.0 + self.eps) * advantages
        loss_clip = th.min(ratio * advantages, clipped).mean()

        return loss_clip + loss_entropy

    def update(self, dataset: dict):
        '''Train the actor.'''

        dataset_size = dataset['states'].shape[0]
    
        for epoch in range(self.epochs):
            # shuffle the start of each epoch
            permutation = th.randperm(dataset_size)
        
            # process entire dataset
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = permutation[start_idx:end_idx]
            
                # update
                loss = -self.objective(dataset, batch_indices)
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
