"""
Expert trajectory collection and management utilities.

This module contains utilities for collecting, processing, and managing expert trajectories
that are used for imitation learning.
"""
import os
import numpy as np
import pickle
import torch
from tqdm import tqdm


def collect_expert_trajectories(policy, env, n_episodes=10, render=False, deterministic=True):
    """
    Collect expert trajectories from a given policy in an environment.
    
    Args:
        policy: The expert policy to collect trajectories from
        env: The environment to interact with
        n_episodes (int): Number of episodes to collect
        render (bool): Whether to render the environment
        deterministic (bool): Whether to use deterministic actions
        
    Returns:
        Dictionary containing collected states, actions, rewards, dones, and episode stats
    """
    trajectories = {
        'states': [],
        'actions': [],
        'rewards': [],
        'dones': [],
        'next_states': []
    }
    
    episode_returns = []
    episode_lengths = []
    
    for i in range(n_episodes):
        states, actions, rewards, dones, next_states = [], [], [], [], []
        state, _ = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            if render:
                env.render()
                
            # Get action from policy
            if hasattr(policy, 'get_action'):
                action, _, _ = policy.get_action(state, deterministic=deterministic)
            else:
                # Assume it's a stable-baselines3 policy
                action, _ = policy.predict(state, deterministic=deterministic)
                
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)
            
            # Update for next iteration
            state = next_state
            episode_return += reward
            episode_length += 1
        
        # Store episode data
        trajectories['states'].extend(states)
        trajectories['actions'].extend(actions)
        trajectories['rewards'].extend(rewards)
        trajectories['dones'].extend(dones)
        trajectories['next_states'].extend(next_states)
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode {i+1}/{n_episodes} - Return: {episode_return:.2f}, Length: {episode_length}")
    
    # Add episode statistics
    trajectories['episode_returns'] = episode_returns
    trajectories['episode_lengths'] = episode_lengths
    trajectories['mean_return'] = np.mean(episode_returns)
    trajectories['std_return'] = np.std(episode_returns)
    
    return trajectories


def save_expert_trajectories(trajectories, filepath):
    """
    Save expert trajectories to a file.
    
    Args:
        trajectories: Dictionary containing trajectory data
        filepath (str): Path to save the trajectories
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"Expert trajectories saved to {filepath}")


def load_expert_trajectories(filepath):
    """
    Load expert trajectories from a file.
    
    Args:
        filepath (str): Path to the trajectories file
        
    Returns:
        Dictionary containing trajectory data
    """
    with open(filepath, 'rb') as f:
        trajectories = pickle.load(f)
    
    print(f"Loaded {len(trajectories['states'])} expert transitions from {filepath}")
    print(f"Average expert return: {trajectories['mean_return']:.2f} ± {trajectories['std_return']:.2f}")
    
    return trajectories


def preprocess_expert_trajectories(trajectories, state_normalizer=None, action_normalizer=None):
    """
    Preprocess expert trajectories by normalizing states and actions.
    
    Args:
        trajectories: Dictionary containing trajectory data
        state_normalizer: Function to normalize states
        action_normalizer: Function to normalize actions
        
    Returns:
        Dictionary containing preprocessed trajectory data
    """
    processed = {k: v for k, v in trajectories.items()}
    
    # Normalize states if normalizer is provided
    if state_normalizer is not None:
        processed['states'] = [state_normalizer(s) for s in tqdm(trajectories['states'], desc="Normalizing states")]
        if 'next_states' in trajectories:
            processed['next_states'] = [state_normalizer(s) for s in tqdm(trajectories['next_states'], desc="Normalizing next states")]
    
    # Normalize actions if normalizer is provided
    if action_normalizer is not None:
        processed['actions'] = [action_normalizer(a) for a in tqdm(trajectories['actions'], desc="Normalizing actions")]
    
    return processed


def create_trajectory_dataset(trajectories, batch_size=64, shuffle=True, device='cuda'):
    """
    Create a PyTorch dataset from trajectories for easier training.
    
    Args:
        trajectories: Dictionary containing trajectory data
        batch_size (int): Batch size for the dataset
        shuffle (bool): Whether to shuffle the data
        device (str): Device to load the data on
        
    Returns:
        DataLoader containing the trajectory data
    """
    states = torch.FloatTensor(np.array(trajectories['states']))
    actions = torch.FloatTensor(np.array(trajectories['actions']))
    
    # Create TensorDataset and DataLoader
    dataset = torch.utils.data.TensorDataset(states, actions)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader