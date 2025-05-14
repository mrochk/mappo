import abc
import torch as th
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class Critic(nn.Module, abc.ABC):
    '''PPO Base Critic Module'''

    def __init__(self, statesize: int, batch_size: int = 64, epochs: int = 10, lr: float = 0.0001):
        '''
        Args:
            statesize (int): State space size (size of first layer input).
            batch_size (int, optional): Batch size for training. Defaults to 64.
            epochs (int): How many times we update when training.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.0003.
        '''
        super().__init__()
        self.ssize, self.epochs, self.batch_size = statesize, epochs, batch_size

        self.value_net_in = nn.Sequential(
            nn.Linear(statesize, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU()
        )

        self.value_net_out = nn.Linear(512, 1)

        # the separation in value_net_int and value_net_out is made to mimic the sb3 implementation
        # such that we can then load pretrained weights from agents trained with sb3 

        self.optim = th.optim.Adam(self.parameters(), lr)

    def compute_gae(self, rewards: Tensor, values: Tensor, last_value: Tensor, 
                        gamma: float, lam: float) -> tuple[Tensor, Tensor]:
        """Compute generalized advantage estimation (GAE) for a given episode.
    
        Args:
            rewards: tensor containing rewards
            values: tensor containing value estimates
            last_value: value estimate for last timestep
            gamma: discount factor
            lam: lambda 
        
        Returns:
            advantages: tensor containing advantages
            returns: tensor containing returns = advantages + values
        """
        horizon = len(rewards)
        advantages, returns = th.zeros_like(rewards), th.zeros_like(rewards)
    
        next_value = last_value
        next_advantage = 0.0
    
        # compute advantages and returns in backward
        for t in reversed(range(horizon)):

            # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
        
            # GAE = A_t = delta_t + (gamma * lambda) * A_{t+1}
            advantages[t] = delta + gamma * lam * next_advantage
        
            next_advantage = advantages[t]
            next_value = values[t]
        
            # returns = advantages + values
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

class DecentralizedCritic(Critic):
    '''Decentralized PPO Critic Module. Used in `IPPO`.'''

    def forward(self, state: th.Tensor) -> th.Tensor:
        out = self.value_net_in(state)
        return self.value_net_out(out)

    def value(self, state: th.Tensor):
        state = state.flatten(start_dim=-2)
        return self.forward(state)

    def update(self, dataset_agent: dict):
        avgloss = 0.0
        n_batches = 0
    
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
        
            states = dataset_agent['states']
            returns = dataset_agent['returns']
            
            dataset_size = states.shape[0]
            
            # Shuffle the entire dataset at the start of each epoch
            permutation = th.randperm(dataset_size)
            
            # Process the entire dataset in mini-batches
            for start_idx in range(0, dataset_size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, dataset_size)
                batch_indices = permutation[start_idx:end_idx]
                
                # Get mini-batch data
                batch_states = states[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Make predictions and compute loss
                predictions = self.value(batch_states).squeeze(1)
                loss = F.mse_loss(predictions, batch_returns)
                
                # Update model
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                
                # Track loss
                epoch_loss += loss.item()
                epoch_batches += 1
        
            # Accumulate total loss and batches
            avgloss += epoch_loss
            n_batches += epoch_batches
    
        # Print average loss (total loss / total number of batches)
        return avgloss / n_batches

    def add_gae(self, agent_trajectories: list, gamma: float = 0.99, lam: float = 0.95) -> dict:
        # Process each agent's trajectories
        for trajectory in agent_trajectories:
            states = trajectory['states']
            rewards = trajectory['rewards']
            
            # Compute values for all states in the trajectory
            with th.no_grad():
                values = self.value(states).squeeze(-1)
                
                # For the last state, we need to estimate its value for bootstrapping
                # If this is a terminal state, last_value should be 0
                # However, in practice, we're going to use the critic's estimate
                last_value = th.zeros(1)
                if len(states) > 0:
                    last_value = self.value(states[-1:]).squeeze(-1)
                
            # Compute advantages and returns
            advantages, returns = self.compute_gae(rewards, values, last_value, gamma, lam)
            
            # Add to trajectory
            trajectory['advantages'] = advantages
            trajectory['rtgs'] = returns
    
        return agent_trajectories

class CentralizedCritic(Critic):
    '''Centralized PPO Critic Module. Used in `MAPPO`.'''

    def __init__(self, statesize: int, batch_size: int = 64, epochs: int = 10, lr: float = 0.0001):
        '''
        Args:
            statesize (int): State space size (size of first layer input).
            batch_size (int, optional): Batch size for training. Defaults to 64.
            epochs (int): How many times we update when training.
            lr (float, optional): Learning rate for Adam optimizer. Defaults to 0.0003.
        '''
        super().__init__(statesize, batch_size, epochs, lr)

        # added layer to include other agent's action
        self.middle = nn.Sequential(nn.Linear(512 + 6, 512), nn.ReLU()) 

    def forward(self, state: Tensor, other_action: Tensor) -> Tensor:
        out = self.value_net_in(state)

        # include other agent's action
        out = th.cat([out, other_action], dim=-1)

        # process concatenated state
        out = self.middle(out)
        return self.value_net_out(out)

    def value(self, state: Tensor, other_action: Tensor, n_actions: int = 6):
        # convert to one-hot
        masked_action = other_action.clone()
        masked_action[masked_action == -1] = 0
        one_hot = F.one_hot(masked_action, num_classes=n_actions).float()
        one_hot[other_action == -1] = 0.0

        # process state, other_action
        state = state.flatten(start_dim=-2)
        return self.forward(state, one_hot)

    def update(self, dataset: dict):

        avgloss = 0.0
        n_batches = 0
    
        for epoch in range(self.epochs):
            epoch_loss = 0.0
            epoch_batches = 0
        
            for agent in dataset.keys():
                data = dataset[agent]
                states = data['states']
                other_actions = data['other_actions']
                returns = data['returns']
            
                dataset_size = states.shape[0]
            
                # Shuffle the entire dataset at the start of each epoch
                permutation = th.randperm(dataset_size)
            
                # Process the entire dataset in mini-batches
                for start_idx in range(0, dataset_size, self.batch_size):
                    end_idx = min(start_idx + self.batch_size, dataset_size)
                    batch_indices = permutation[start_idx:end_idx]
                
                    # Get mini-batch data
                    batch_states = states[batch_indices]
                    batch_other_actions = other_actions[batch_indices]
                    batch_returns = returns[batch_indices]
                
                    # Make predictions and compute loss
                    predictions = self.value(batch_states, batch_other_actions).squeeze(1)
                    loss = F.mse_loss(predictions, batch_returns)
                
                    # Update model
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()
                
                    # Track loss
                    epoch_loss += loss.item()
                    epoch_batches += 1
        
            # Accumulate total loss and batches
            avgloss += epoch_loss
            n_batches += epoch_batches
    
        # Print average loss (total loss / total number of batches)
        return avgloss / n_batches

    def add_gae(self, trajectories: dict, gamma: float = 0.99, lam: float = 0.95) -> dict:
        # Process each agent's trajectories
        for agent, agent_trajectories in trajectories.items():
            for trajectory in agent_trajectories:
                states = trajectory['states']
                other_actions = trajectory['other_actions']
                rewards = trajectory['rewards']
            
                # Compute values for all states in the trajectory
                with th.no_grad():
                    values = self.value(states, other_actions).squeeze(-1)
                
                    # For the last state, we need to estimate its value for bootstrapping
                    # If this is a terminal state, last_value should be 0
                    # However, in practice, we're going to use the critic's estimate
                    last_value = th.zeros(1)
                    if len(states) > 0:
                        last_value = self.value(states[-1:], other_actions[-1:]).squeeze(-1)
                
                # Compute advantages and returns
                advantages, returns = self.compute_gae(rewards, values, last_value, gamma, lam)
            
                # Add to trajectory
                trajectory['advantages'] = advantages
                trajectory['rtgs'] = returns
    
        return trajectories
