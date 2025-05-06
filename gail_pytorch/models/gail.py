"""
GAIL (Generative Adversarial Imitation Learning) implementation.

This file contains the core implementation of GAIL algorithm.
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


class Discriminator(nn.Module):
    """
    Discriminator network for GAIL.
    
    Distinguishes between expert trajectories and agent trajectories.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dims=(256, 256), device='cuda'):
        """
        Initialize the discriminator network.
        
        Args:
            state_dim (int): Dimension of the state space
            action_dim (int): Dimension of the action space
            hidden_dims (tuple): Dimensions of hidden layers
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        super(Discriminator, self).__init__()
        
        self.device = device
        
        # Create network architecture
        layers = []
        dims = [state_dim + action_dim] + list(hidden_dims) + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        
        self.model = nn.Sequential(*layers)
        self.to(device)
    
    def forward(self, states, actions):
        """Forward pass through the discriminator."""
        inputs = torch.cat([states, actions], dim=1)
        return torch.sigmoid(self.model(inputs))


class GAIL:
    """
    Generative Adversarial Imitation Learning.
    
    Implements the GAIL algorithm as described in the paper:
    "Generative Adversarial Imitation Learning" by Ho & Ermon, 2016.
    """
    
    def __init__(
        self,
        policy,
        expert_trajectories,
        discriminator_lr=1e-3,
        gamma=0.99,
        batch_size=64,
        entropy_weight=1e-2,
        clip_grad_norm=None,
        log_dir='./data/logs',
        device='cuda'
    ):
        """
        Initialize the GAIL algorithm.
        
        Args:
            policy: Policy network to be trained
            expert_trajectories: Expert demonstrations
            discriminator_lr (float): Learning rate for the discriminator
            gamma (float): Discount factor
            batch_size (int): Batch size for discriminator training
            entropy_weight (float): Weight for entropy regularization
            clip_grad_norm (float): Gradient clipping norm
            log_dir (str): Directory for tensorboard logs
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.policy = policy
        self.expert_trajectories = expert_trajectories
        self.device = device if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        
        # Verify if CUDA is available
        if device == 'cuda' and not torch.cuda.is_available():
            print("CUDA is not available. Using CPU instead.")
        
        # Get state and action dimensions from expert trajectories
        if len(expert_trajectories['states']) > 0:
            state_sample = expert_trajectories['states'][0]
            action_sample = expert_trajectories['actions'][0]
            state_dim = state_sample.shape[0] if len(state_sample.shape) > 0 else 1
            action_dim = action_sample.shape[0] if len(action_sample.shape) > 0 else 1
        else:
            raise ValueError("Expert trajectories are empty.")
        
        # Create discriminator
        self.discriminator = Discriminator(state_dim, action_dim, device=self.device)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=discriminator_lr)
        
        # Other parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.entropy_weight = entropy_weight
        self.clip_grad_norm = clip_grad_norm
        
        # Setup logging
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.log_dir = log_dir
        
        # Counters
        self.iterations = 0
    
    def get_reward(self, states, actions):
        """
        Calculate the GAIL reward for a given state-action pair.
        
        Args:
            states: States from the environment
            actions: Actions from the policy
            
        Returns:
            Rewards calculated using the discriminator
        """
        with torch.no_grad():
            states_tensor = torch.FloatTensor(states).to(self.device)
            actions_tensor = torch.FloatTensor(actions).to(self.device)
            
            # 確保動作張量具有正確的維度
            # 對於離散動作，需要處理一維張量
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(1)
            
            # 確保狀態和動作張量具有相同的批次維度
            if states_tensor.dim() == 1:
                states_tensor = states_tensor.unsqueeze(0)
            if actions_tensor.dim() == 1:
                actions_tensor = actions_tensor.unsqueeze(0)
            
            d_output = self.discriminator(states_tensor, actions_tensor)
            # -log(1-D) as the reward
            rewards = -torch.log(1 - d_output + 1e-8).detach().cpu().numpy()
            
            # 確保始終返回數組，即使是單個值
            if rewards.size == 1:
                return np.array([rewards.item()])
            else:
                return rewards.flatten()  # 使用flatten確保返回一維數組
    
    def update_discriminator(self, agent_states, agent_actions):
        """
        Update the discriminator using a batch of agent and expert data.
        
        Args:
            agent_states: States from the agent
            agent_actions: Actions from the agent
            
        Returns:
            Discriminator loss value
        """
        # 確保我們有足夠的數據可以使用
        if len(agent_states) < self.batch_size:
            # 如果數據不足，複製現有數據直到達到批次大小
            factor = int(np.ceil(self.batch_size / len(agent_states)))
            agent_states = np.tile(agent_states, (factor, 1))[:self.batch_size]
            
            # 處理離散動作（可能是標量）和連續動作（可能是數組）
            if isinstance(agent_actions, (int, float)) or (hasattr(agent_actions, 'shape') and len(agent_actions.shape) == 0):
                agent_actions = np.array([agent_actions] * self.batch_size)
            else:
                if len(np.array(agent_actions).shape) == 1:
                    agent_actions = np.tile(agent_actions, factor)[:self.batch_size]
                else:
                    agent_actions = np.tile(agent_actions, (factor, 1))[:self.batch_size]
                
        # 確保維度正確
        agent_states = np.array(agent_states)
        agent_actions = np.array(agent_actions)
        
        # 轉換為PyTorch張量
        agent_states = torch.FloatTensor(agent_states).to(self.device)
        
        # 確保動作是浮點型張量，並處理離散動作
        if len(agent_actions.shape) == 1:
            agent_actions = torch.FloatTensor(agent_actions).unsqueeze(1).to(self.device)
        else:
            agent_actions = torch.FloatTensor(agent_actions).to(self.device)
        
        # 抽樣專家數據
        try:
            # 安全獲取專家軌跡長度
            expert_data_length = len(self.expert_trajectories['states'])
            indices = np.random.randint(0, expert_data_length, self.batch_size)
            
            # 獲取專家狀態
            expert_states = []
            for i in indices:
                expert_states.append(self.expert_trajectories['states'][i])
            expert_states = torch.FloatTensor(np.array(expert_states)).to(self.device)
            
            # 獲取專家動作 - 處理離散和連續動作
            expert_actions = []
            for i in indices:
                action = self.expert_trajectories['actions'][i]
                expert_actions.append(action)
            
            # 將專家動作轉換為適當的張量格式
            expert_actions = np.array(expert_actions)
            if len(expert_actions.shape) == 1:
                expert_actions = torch.FloatTensor(expert_actions).unsqueeze(1).to(self.device)
            else:
                expert_actions = torch.FloatTensor(expert_actions).to(self.device)
                
        except (TypeError, IndexError) as e:
            print(f"Error processing expert data: {e}")
            print(f"Expert trajectory format: {type(self.expert_trajectories)}")
            print(f"States shape: {np.array(self.expert_trajectories['states']).shape}")
            print(f"Actions shape: {np.array(self.expert_trajectories['actions']).shape}")
            raise
        
        # 訓練判別器
        self.disc_optimizer.zero_grad()
        
        # 預測專家數據（應輸出1）
        expert_preds = self.discriminator(expert_states, expert_actions)
        expert_loss = nn.BCELoss()(
            expert_preds, 
            torch.ones((self.batch_size, 1), device=self.device)
        )
        
        # 預測智能體數據（應輸出0）
        agent_preds = self.discriminator(agent_states, agent_actions)
        agent_loss = nn.BCELoss()(
            agent_preds, 
            torch.zeros((self.batch_size, 1), device=self.device)
        )
        
        # 總損失和反向傳播
        disc_loss = expert_loss + agent_loss
        disc_loss.backward()
        
        # 可選梯度裁剪
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(), 
                self.clip_grad_norm
            )
        
        self.disc_optimizer.step()
        
        # 記錄指標
        self.iterations += 1
        self.writer.add_scalar('discriminator/loss', disc_loss.item(), self.iterations)
        self.writer.add_scalar('discriminator/expert_acc', 
                             (expert_preds > 0.5).float().mean().item(), 
                             self.iterations)
        self.writer.add_scalar('discriminator/agent_acc', 
                             (agent_preds < 0.5).float().mean().item(), 
                             self.iterations)
        
        return disc_loss.item()
    
    def save(self, path=None):
        """Save the GAIL model."""
        if path is None:
            path = os.path.join(self.log_dir, f'gail_model_{self.iterations}.pt')
        
        # 保存完整模型（包括判別器和策略）
        torch.save({
            'discriminator': self.discriminator.state_dict(),
            'disc_optimizer': self.disc_optimizer.state_dict(),
            'policy': self.policy.state_dict(),
            'iterations': self.iterations
        }, path)
        
        # 單獨保存策略模型，以便後續評估
        policy_path = path.replace('gail_model', 'policy_model')
        torch.save(self.policy.state_dict(), policy_path)
        print(f"Policy saved to {policy_path}")
        
        return path
    
    def load(self, path):
        """Load the GAIL model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.discriminator.load_state_dict(checkpoint['discriminator'])
        self.disc_optimizer.load_state_dict(checkpoint['disc_optimizer'])
        self.iterations = checkpoint['iterations']