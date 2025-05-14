import torch as th
from torch import nn
from torch.nn import functional as F

class Critic(nn.Module):
    '''PPO Critic network module.'''

    def __init__(
        self, 
        state_space_size: int, 
        batch_size: int = 64,
        epochs: int = 10,
    ):

        super().__init__()

        self.ss_size = state_space_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.value_net_in = nn.Sequential(
            nn.Linear(state_space_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.middle = nn.Sequential(
            nn.Linear(512 + 6, 512),
            nn.ReLU(),
        ) 

        self.value_net_out = nn.Linear(512, 1)

        self.optim = th.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, state: th.Tensor, other_action: th.Tensor) -> th.Tensor:
        out = self.value_net_in(state)
        out = th.cat([out, other_action], dim=-1)
        out = self.middle(out)
        return self.value_net_out(out)

    def value(self, state: th.Tensor, other_action: th.Tensor):
        n_actions = 6 
        masked_action = other_action.clone()
        masked_action[masked_action == -1] = 0
        one_hot = F.one_hot(masked_action, num_classes=n_actions).float()
        one_hot[other_action == -1] = 0.0
        state = state.flatten(start_dim=-2)
        return self.forward(state, one_hot)

    def update(self, dataset: dict):
        """Update the critic using mini-batches covering the full dataset.
    
        Args:
            dataset: Dictionary containing data for each agent with states, actions, returns, etc.
        """
        losstrack = 0.0
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
            losstrack += epoch_loss
            n_batches += epoch_batches
    
        # Print average loss (total loss / total number of batches)
        return losstrack / n_batches

    def compute_gae(self, rewards: th.Tensor, values: th.Tensor, last_value: th.Tensor, 
                gamma: float, lam: float) -> tuple[th.Tensor, th.Tensor]:
        """Compute generalized advantage estimation.
    
        Args:
            rewards: tensor of shape (T) containing rewards
            values: tensor of shape (T) containing value estimates
            last_value: scalar tensor containing the value estimate at T+1
            gamma: discount factor
            lam: lambda for GAE
        
        Returns:
            advantages: tensor of shape (T) containing advantages
            returns: tensor of shape (T) containing returns (advantages + values)
        """
        T = len(rewards)
        advantages = th.zeros_like(rewards)
        returns = th.zeros_like(rewards)
    
        # Initialize with the value of the last state
        next_value = last_value
        next_advantage = 0
    
        # Compute advantages and returns in a backward loop
        for t in reversed(range(T)):
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
        
            # GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
            advantages[t] = delta + gamma * lam * next_advantage
        
            # Update for next iteration
            next_advantage = advantages[t]
            next_value = values[t]
        
            # Returns = advantages + values
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

    def add_gae(self, trajectories: dict, gamma: float = 0.99, lam: float = 0.95) -> dict:
        """Add generalized advantage estimation to trajectories.
    
        Args:
            trajectories: dict containing trajectories for each agent
            gamma: discount factor
            lam: lambda for GAE
        
        Returns:
            trajectories with added 'advantages' and 'rtgs' fields
        """
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

class CriticIPPO(nn.Module):
    '''PPO Critic network module.'''

    def __init__(
        self, 
        state_space_size: int, 
        batch_size: int = 64,
        epochs: int = 10,
    ):

        super().__init__()

        self.ss_size = state_space_size
        self.batch_size = batch_size
        self.epochs = epochs

        self.value_net_in = nn.Sequential(
            nn.Linear(state_space_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

        self.value_net_out = nn.Linear(512, 1)

        self.optim = th.optim.Adam(self.parameters(), lr=0.00001)

    def forward(self, state: th.Tensor) -> th.Tensor:
        out = self.value_net_in(state)
        return self.value_net_out(out)

    def value(self, state: th.Tensor):
        state = state.flatten(start_dim=-2)
        return self.forward(state)

    def update(self, dataset_agent: dict):
        """Update the critic using mini-batches covering the full dataset.
    
        Args:
            dataset: Dictionary containing data for each agent with states, actions, returns, etc.
        """
        losstrack = 0.0
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
            losstrack += epoch_loss
            n_batches += epoch_batches
    
        # Print average loss (total loss / total number of batches)
        return losstrack / n_batches

    def compute_gae(self, rewards: th.Tensor, values: th.Tensor, last_value: th.Tensor, 
                gamma: float, lam: float) -> tuple[th.Tensor, th.Tensor]:
        """Compute generalized advantage estimation.
    
        Args:
            rewards: tensor of shape (T) containing rewards
            values: tensor of shape (T) containing value estimates
            last_value: scalar tensor containing the value estimate at T+1
            gamma: discount factor
            lam: lambda for GAE
        
        Returns:
            advantages: tensor of shape (T) containing advantages
            returns: tensor of shape (T) containing returns (advantages + values)
        """
        T = len(rewards)
        advantages = th.zeros_like(rewards)
        returns = th.zeros_like(rewards)
    
        # Initialize with the value of the last state
        next_value = last_value
        next_advantage = 0
    
        # Compute advantages and returns in a backward loop
        for t in reversed(range(T)):
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[t] + gamma * next_value - values[t]
        
            # GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
            advantages[t] = delta + gamma * lam * next_advantage
        
            # Update for next iteration
            next_advantage = advantages[t]
            next_value = values[t]
        
            # Returns = advantages + values
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns

    def add_gae(self, agent_trajectories: list, gamma: float = 0.99, lam: float = 0.95) -> dict:
        """Add generalized advantage estimation to trajectories.
    
        Args:
            trajectories: dict containing trajectories for each agent
            gamma: discount factor
            lam: lambda for GAE
        
        Returns:
            trajectories with added 'advantages' and 'rtgs' fields
        """
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