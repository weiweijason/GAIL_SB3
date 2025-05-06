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
    
    return parser.parse_args()


def evaluate_policy(policy, env, n_episodes=10, render=False):
    """Evaluate the policy on the environment."""
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
    
    # Load expert trajectories
    expert_trajectories = load_expert_trajectories(args.expert_data)
    
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
    eval_return, eval_length = evaluate_policy(policy, env, args.n_eval_episodes)
    writer.add_scalar('eval/return', eval_return, timesteps_so_far)
    writer.add_scalar('eval/length', eval_length, timesteps_so_far)
    print(f"Initial evaluation: Mean return = {eval_return:.2f}, Mean length = {eval_length:.2f}")
    
    while timesteps_so_far < args.total_timesteps:
        # Reset for new episode
        state, _ = env.reset()
        done = False
        episode_length = 0
        episode_return = 0
        episode_disc_rewards = 0
        
        # Collect experience for policy update
        states = []
        actions = []
        log_probs = []
        values = []
        rewards = []
        masks = []
        
        while not done and episode_length < args.max_ep_len:
            # Get action from policy
            if is_discrete:
                action, log_prob, value = policy.get_action(state)
            else:
                action, log_prob, value = policy.get_action(state)
            
            # Store experience
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            
            # Take step in environment
            next_state, env_reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            mask = 1.0 - float(done)
            
            # Calculate reward using discriminator
            disc_reward = gail.get_reward(np.array([state]), np.array([action]))[0]
            rewards.append(disc_reward)
            masks.append(mask)
            
            # Update tracking variables
            state = next_state
            episode_length += 1
            episode_return += env_reward
            episode_disc_rewards += disc_reward
            timesteps_so_far += 1
            
            # Update discriminator if it's time
            if timesteps_so_far % args.disc_update_freq == 0:
                # Collect recent experience
                recent_states = np.array(states[-args.batch_size:])
                recent_actions = np.array(actions[-args.batch_size:])
                
                # Update discriminator
                disc_loss = gail.update_discriminator(recent_states, recent_actions)
                writer.add_scalar('train/disc_loss', disc_loss, timesteps_so_far)
            
            # Save model if it's time
            if timesteps_so_far % args.save_freq == 0:
                save_path = gail.save(f"{args.log_dir}/gail_model_{timesteps_so_far}.pt")
                print(f"Model saved to {save_path}")
            
            # Evaluate policy if it's time
            if timesteps_so_far % args.eval_freq == 0:
                eval_return, eval_length = evaluate_policy(policy, env, args.n_eval_episodes)
                writer.add_scalar('eval/return', eval_return, timesteps_so_far)
                writer.add_scalar('eval/length', eval_length, timesteps_so_far)
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
    eval_return, eval_length = evaluate_policy(policy, env, args.n_eval_episodes)
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