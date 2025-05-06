"""
Policy network implementations for GAIL.

This file contains implementations of various policy networks that can be used with GAIL.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


class DiscretePolicy(nn.Module):
    """
    Discrete policy network for environments with discrete action spaces.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64), device='cuda'):
        """
        Initialize the discrete policy network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Number of discrete actions
            hidden_dims (tuple): Dimensions of hidden layers
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        super(DiscretePolicy, self).__init__()
        
        self.device = device
        
        # Build network
        layers = []
        dims = [state_dim] + list(hidden_dims)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        self.backbone = nn.Sequential(*layers)
        self.action_head = nn.Linear(hidden_dims[-1], action_dim)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
        self.to(device)
    
    def forward(self, states):
        """Forward pass through the policy network."""
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
            
        features = self.backbone(states)
        action_logits = self.action_head(features)
        state_values = self.value_head(features)
        
        return action_logits, state_values
    
    def get_action(self, state, deterministic=False):
        """
        Get an action from the policy for a given state.
        
        Args:
            state: The current state
            deterministic (bool): If True, return the most probable action
            
        Returns:
            Action index, log probability of the action, and state value
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
            action_logits, state_value = self.forward(state)
            action_probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(action_probs, dim=-1)
            else:
                dist = Categorical(action_probs)
                action = dist.sample()
            
            log_prob = F.log_softmax(action_logits, dim=-1).gather(1, action.unsqueeze(-1))
            
            return action.item(), log_prob.item(), state_value.item()


class ContinuousPolicy(nn.Module):
    """
    Continuous policy network for environments with continuous action spaces.
    Implements a Gaussian policy with state-dependent mean and standard deviation.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=(64, 64), device='cuda'):
        """
        Initialize the continuous policy network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (tuple): Dimensions of hidden layers
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        super(ContinuousPolicy, self).__init__()
        
        self.device = device
        self.action_dim = action_dim
        
        # Build network
        layers = []
        dims = [state_dim] + list(hidden_dims)
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        
        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
        
        # Initialize action head with small weights
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        
        self.to(device)
    
    def forward(self, states):
        """Forward pass through the policy network."""
        if not isinstance(states, torch.Tensor):
            states = torch.FloatTensor(states).to(self.device)
            
        features = self.backbone(states)
        action_means = self.mean_head(features)
        action_log_stds = self.log_std_head(features)
        
        # Constrain log_stds to a reasonable range
        action_log_stds = torch.clamp(action_log_stds, -20, 2)
        
        state_values = self.value_head(features)
        
        return action_means, action_log_stds, state_values
    
    def get_action(self, state, deterministic=False):
        """
        Get an action from the policy for a given state.
        
        Args:
            state: The current state
            deterministic (bool): If True, return the mean action without noise
            
        Returns:
            Action vector, log probability of the action, and state value
        """
        with torch.no_grad():
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
            action_means, action_log_stds, state_value = self.forward(state)
            action_stds = torch.exp(action_log_stds)
            
            if deterministic:
                action = action_means
            else:
                dist = Normal(action_means, action_stds)
                action = dist.sample()
            
            # Calculate log probability
            log_prob = -0.5 * ((action - action_means) / (action_stds + 1e-8)).pow(2) - \
                       action_log_stds - 0.5 * np.log(2 * np.pi)
            log_prob = log_prob.sum(-1, keepdim=True)
            
            return action.cpu().numpy()[0], log_prob.item(), state_value.item()
    
    def evaluate_actions(self, states, actions):
        """
        Evaluate log probabilities and state values for given states and actions.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            
        Returns:
            Log probabilities and state values
        """
        action_means, action_log_stds, state_values = self.forward(states)
        action_stds = torch.exp(action_log_stds)
        
        # Calculate log probability
        log_prob = -0.5 * ((actions - action_means) / (action_stds + 1e-8)).pow(2) - \
                   action_log_stds - 0.5 * np.log(2 * np.pi)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return log_prob, state_values