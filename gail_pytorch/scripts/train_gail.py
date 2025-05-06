"""Training script for GAIL.

This script demonstrates how to train a policy using GAIL with expert demonstrations.
"""
import os
import argparse
import time
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple, List, Dict, Any, Optional

from gail_pytorch.models.gail import GAIL
from gail_pytorch.models.policy import DiscretePolicy, ContinuousPolicy
from gail_pytorch.utils.expert_trajectories import load_expert_trajectories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a policy using GAIL")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym environment name")
    
    # Expert data settings
    parser.add_argument("--expert_data", type=str, required=True,
                        help="Path to expert trajectories file")
    
    # Training settings
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total number of timesteps to train for")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                        help="Maximum episode length")
    parser.add_argument("--policy_update_freq", type=int, default=2048,
                        help="Number of timesteps between policy updates")
    parser.add_argument("--disc_update_freq", type=int, default=1024,
                        help="Number of timesteps between discriminator updates")
    parser.add_argument("--policy_lr", type=float, default=3e-4,
                        help="Policy learning rate")
    parser.add_argument("--disc_lr", type=float, default=3e-4,
                        help="Discriminator learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--entropy_weight", type=float, default=0.01,
                        help="Entropy regularization weight")
    
    # Policy network settings
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden layer dimension for policy and discriminator")
    parser.add_argument("--n_hidden", type=int, default=2,
                        help="Number of hidden layers")
    
    # Logging and saving
    parser.add_argument("--log_dir", type=str, default="./data/logs",
                        help="Directory to save logs")
    parser.add_argument("--save_freq", type=int, default=10000,
                        help="Save frequency in timesteps")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Evaluation frequency in timesteps")
    parser.add_argument("--n_eval_episodes", type=int, default=10,
                        help="Number of episodes for evaluation")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    
    # New arguments for policy optimization
    parser.add_argument("--n_policy_epochs", type=int, default=10,
                        help="Number of policy optimization epochs per update")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda parameter")
    parser.add_argument("--clip_ratio", type=float, default=0.2,
                        help="PPO clip ratio")
    parser.add_argument("--value_coef", type=float, default=0.5,
                        help="Value function coefficient")
    parser.add_argument("--max_grad_norm", type=float, default=0.5,
                        help="Maximum gradient norm for clipping")
    
    # New argument for logging graph
    parser.add_argument("--log_graph", action="store_true",
                        help="Log network architecture graph to TensorBoard")
    
    return parser.parse_args()


def evaluate_policy(policy: torch.nn.Module, env: gym.Env, args, n_episodes: int = 10, 
                   render: bool = False) -> Tuple[float, float]:
    """
    Evaluate the policy on the environment.
    
    Args:
        policy: The policy to evaluate
        env: The environment to evaluate in
        args: Training arguments
        n_episodes: Number of episodes to evaluate
        render: Whether to render the environment
        
    Returns:
        Tuple of (mean_return, mean_episode_length)
    """
    returns = []
    lengths = []
    
    for i in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done and episode_length < args.max_ep_len:
            if render:
                env.render()
            
            # Get action from policy (deterministic at evaluation time)
            if isinstance(policy, (DiscretePolicy, ContinuousPolicy)):
                action, _, _ = policy.get_action(state, deterministic=True)
            else:
                action, _ = policy.predict(state, deterministic=True)
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update for next iteration
            state = next_state
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
    
    mean_return = np.mean(returns)
    mean_length = np.mean(lengths)
    
    return mean_return, mean_length


def compute_gae(rewards: List[float], values: List[float], masks: List[float], 
               next_value: float, gamma: float, lam: float) -> List[float]:
    """
    Compute Generalized Advantage Estimation.
    
    Args:
        rewards: List of rewards
        values: List of state values
        masks: List of masks (0 if episode done, 1 otherwise)
        next_value: Value of the next state
        gamma: Discount factor
        lam: GAE lambda parameter
        
    Returns:
        List of advantages
    """
    advantages = []
    gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_v = next_value
        else:
            next_v = values[t + 1]
            
        delta = rewards[t] + gamma * next_v * masks[t] - values[t]
        gae = delta + gamma * lam * masks[t] * gae
        advantages.insert(0, gae)
        
    return advantages


def update_policy(policy: torch.nn.Module, optimizer: torch.optim.Optimizer, 
                 states: np.ndarray, actions: np.ndarray, log_probs_old: List[float], 
                 advantages: np.ndarray, returns: np.ndarray, args) -> Dict[str, float]:
    """
    Update the policy using PPO.
    
    Args:
        policy: The policy to update
        optimizer: The optimizer to use
        states: The states from the environment
        actions: The actions taken by the policy
        log_probs_old: The log probabilities of the actions under the old policy
        advantages: The advantages for each state-action pair
        returns: The returns for each state
        args: Training arguments
        
    Returns:
        Dictionary of training metrics
    """
    states = torch.FloatTensor(states).to(args.device)
    if isinstance(policy, DiscretePolicy):
        actions = torch.LongTensor(actions).to(args.device)
    else:
        actions = torch.FloatTensor(actions).to(args.device)
    log_probs_old = torch.FloatTensor(log_probs_old).to(args.device)
    advantages = torch.FloatTensor(advantages).to(args.device)
    returns = torch.FloatTensor(returns).to(args.device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    metrics = {
        'policy_loss': 0,
        'value_loss': 0,
        'entropy': 0,
        'approx_kl': 0,
        'clip_fraction': 0
    }
    
    # Run multiple epochs of policy optimization
    for _ in range(args.n_policy_epochs):
        # Get new log probs and values
        if isinstance(policy, DiscretePolicy):
            action_logits, values = policy(states)
            dist = torch.distributions.Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
        else:
            action_means, action_log_stds, values = policy(states)
            action_stds = torch.exp(action_log_stds)
            dist = torch.distributions.Normal(action_means, action_stds)
            log_probs, _ = policy.evaluate_actions(states, actions)
            entropy = dist.entropy().sum(dim=-1).mean()
        
        # PPO policy loss
        ratio = torch.exp(log_probs - log_probs_old)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = values.squeeze()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Total loss
        loss = policy_loss + args.value_coef * value_loss - args.entropy_weight * entropy
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        if args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
        optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            metrics['policy_loss'] += policy_loss.item() / args.n_policy_epochs
            metrics['value_loss'] += value_loss.item() / args.n_policy_epochs
            metrics['entropy'] += entropy.item() / args.n_policy_epochs
            metrics['approx_kl'] += ((log_probs_old - log_probs) ** 2).mean().item() / args.n_policy_epochs
            metrics['clip_fraction'] += ((ratio - 1.0).abs() > args.clip_ratio).float().mean().item() / args.n_policy_epochs
    
    return metrics


def main(args):
    """Main training function."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create environment
    env = gym.make(args.env)
    
    # Determine if environment has discrete or continuous action space
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        action_dim = env.action_space.n
    else:
        is_discrete = False
        action_dim = env.action_space.shape[0]
    
    # Get state dimension
    state_dim = env.observation_space.shape[0]
    
    # Create policy network
    hidden_dims = (args.hidden_dim,) * args.n_hidden
    if is_discrete:
        policy = DiscretePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=args.device
        )
        print(f"Created discrete policy with state_dim={state_dim}, action_dim={action_dim}")
    else:
        policy = ContinuousPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=args.device
        )
        print(f"Created continuous policy with state_dim={state_dim}, action_dim={action_dim}")
    
    # 添加網絡架構圖
    if args.log_graph:
        dummy_input = torch.zeros((1, state_dim)).to(args.device)
        if is_discrete:
            # 對於離散政策，紀錄前向傳遞的計算圖
            writer.add_graph(policy, dummy_input)
    
    # Load expert trajectories
    expert_trajectories = load_expert_trajectories(args.expert_data)
    print(f"Expert mean return: {expert_trajectories['mean_return']:.2f}")
    
    # Create GAIL instance
    gail = GAIL(
        policy=policy,
        expert_trajectories=expert_trajectories,
        discriminator_lr=args.disc_lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        entropy_weight=args.entropy_weight,
        log_dir=args.log_dir,
        device=args.device
    )
    
    # Create optimizer for policy
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    
    # Training loop
    print(f"Starting training for {args.total_timesteps} timesteps")
    
    timesteps_so_far = 0
    episodes_so_far = 0
    start_time = time.time()
    
    # Initial evaluation
    eval_return, eval_length = evaluate_policy(policy, env, args, args.n_eval_episodes)
    writer.add_scalar('eval/return', eval_return, timesteps_so_far)
    writer.add_scalar('eval/length', eval_length, timesteps_so_far)
    print(f"Initial evaluation: Mean return = {eval_return:.2f}, Mean length = {eval_length:.2f}")
    
    # For policy update
    policy_update_states = []
    policy_update_actions = []
    policy_update_log_probs = []
    policy_update_disc_rewards = []
    policy_update_masks = []
    policy_update_values = []
    
    while timesteps_so_far < args.total_timesteps:
        # Reset for new episode
        state, _ = env.reset()
        done = False
        episode_length = 0
        episode_return = 0
        episode_disc_rewards = 0
        last_value = 0
        
        # Collect experience for episode
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_values = []
        episode_rewards = []
        episode_disc_rewards_list = []
        episode_masks = []
        
        while not done and episode_length < args.max_ep_len:
            # Get action from policy
            if is_discrete:
                action, log_prob, value = policy.get_action(state)
            else:
                action, log_prob, value = policy.get_action(state)
            
            # Store experience
            episode_states.append(state)
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_values.append(value)
            
            # Take step in environment
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            mask = 1.0 - float(done)
            
            # Calculate reward using discriminator
            disc_reward = gail.get_reward(np.array([state]), np.array([action]))[0]
            episode_rewards.append(env_reward)  # Store environment reward for logging
            episode_disc_rewards_list.append(disc_reward)  # Store discriminator reward for training
            episode_masks.append(mask)
            
            # Save for policy update
            policy_update_states.append(state)
            policy_update_actions.append(action)
            policy_update_log_probs.append(log_prob)
            policy_update_disc_rewards.append(disc_reward)
            policy_update_masks.append(mask)
            policy_update_values.append(value)
            
            # Update tracking variables
            state = next_state
            episode_length += 1
            episode_return += env_reward
            episode_disc_rewards += disc_reward
            last_value = value
            timesteps_so_far += 1
            
            # Update discriminator if it's time
            if timesteps_so_far % args.disc_update_freq == 0 and len(policy_update_states) >= args.batch_size:
                # Collect recent experience
                recent_states = np.array(policy_update_states[-args.batch_size:])
                recent_actions = np.array(policy_update_actions[-args.batch_size:])
                
                # Update discriminator
                disc_loss = gail.update_discriminator(recent_states, recent_actions)
                writer.add_scalar('train/disc_loss', disc_loss, timesteps_so_far)
            
            # Update policy if it's time
            if timesteps_so_far % args.policy_update_freq == 0 and len(policy_update_states) >= args.batch_size:
                # Compute advantages and returns
                if done:
                    next_value = 0
                else:
                    # Get value of next state
                    with torch.no_grad():
                        if is_discrete:
                            _, next_value = policy(torch.FloatTensor(np.array([state])).to(args.device))
                        else:
                            _, _, next_value = policy(torch.FloatTensor(np.array([state])).to(args.device))
                        next_value = next_value.item()
                
                # Compute GAE advantages and returns
                advantages = compute_gae(
                    policy_update_disc_rewards, policy_update_values, policy_update_masks,
                    next_value, args.gamma, args.gae_lambda
                )
                returns = [adv + val for adv, val in zip(advantages, policy_update_values)]
                
                # Update policy
                metrics = update_policy(
                    policy, policy_optimizer,
                    np.array(policy_update_states), np.array(policy_update_actions),
                    policy_update_log_probs, np.array(advantages), np.array(returns),
                    args
                )
                
                # Log policy metrics
                for k, v in metrics.items():
                    writer.add_scalar(f'train/{k}', v, timesteps_so_far)
                
                # Clear buffers after update
                policy_update_states = []
                policy_update_actions = []
                policy_update_log_probs = []
                policy_update_disc_rewards = []
                policy_update_masks = []
                policy_update_values = []
            
            # Save model if it's time
            if timesteps_so_far % args.save_freq == 0:
                save_path = gail.save(f"{args.log_dir}/gail_model_{timesteps_so_far}.pt")
                print(f"Model saved to {save_path}")
            
            # Evaluate policy if it's time
            if timesteps_so_far % args.eval_freq == 0:
                eval_return, eval_length = evaluate_policy(policy, env, args, args.n_eval_episodes)
                writer.add_scalar('eval/return', eval_return, timesteps_so_far)
                writer.add_scalar('eval/length', eval_length, timesteps_so_far)
                
                # 添加政策網絡參數分佈
                for name, param in policy.named_parameters():
                    writer.add_histogram(f'parameters/{name}', param.data, timesteps_so_far)
                    if param.grad is not None:
                        writer.add_histogram(f'gradients/{name}', param.grad.data, timesteps_so_far)
                
                print(f"Evaluation at {timesteps_so_far} timesteps: Mean return = {eval_return:.2f}, Mean length = {eval_length:.2f}")
        
        # Episode is done
        episodes_so_far += 1
        
        # Log episode stats
        writer.add_scalar('train/episode_length', episode_length, episodes_so_far)
        writer.add_scalar('train/episode_return', episode_return, episodes_so_far)
        writer.add_scalar('train/episode_disc_reward', episode_disc_rewards, episodes_so_far)
        
        # Print progress
        print(f"Episode {episodes_so_far} - Length: {episode_length}, Return: {episode_return:.2f}, Disc Reward: {episode_disc_rewards:.2f}")
    
    # Final evaluation
    eval_return, eval_length = evaluate_policy(policy, env, args, args.n_eval_episodes)
    writer.add_scalar('eval/return', eval_return, timesteps_so_far)
    writer.add_scalar('eval/length', eval_length, timesteps_so_far)
    print(f"Final evaluation: Mean return = {eval_return:.2f}, Mean length = {eval_length:.2f}")
    
    # Save final model
    final_save_path = gail.save(f"{args.log_dir}/gail_model_final.pt")
    print(f"Final model saved to {final_save_path}")
    
    # Close environment and writer
    env.close()
    writer.close()
    
    # Print training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")


if __name__ == "__main__":
    args = parse_args()
    main(args)