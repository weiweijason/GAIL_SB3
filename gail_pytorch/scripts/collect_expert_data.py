"""Script to collect expert trajectories from a pre-trained expert.

This script allows you to collect expert trajectories from a pre-trained policy
(either a custom policy or one from stable-baselines3).
"""
import os
import argparse
import gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3

from gail_pytorch.utils.expert_trajectories import collect_expert_trajectories, save_expert_trajectories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Collect expert trajectories")
    
    # Environment settings
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym environment name")
    
    # Expert settings
    parser.add_argument("--expert_path", type=str, required=True,
                        help="Path to expert model file")
    parser.add_argument("--expert_algo", type=str, default="ppo", choices=["ppo", "sac", "td3"],
                        help="Algorithm used to train the expert")
    
    # Collection settings
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of episodes to collect")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                        help="Maximum episode length")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use deterministic actions when collecting trajectories")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment during collection")
    
    # Output settings
    parser.add_argument("--output_path", type=str, default="./data/expert_trajectories/expert_data.pkl",
                        help="Path to save the collected trajectories")
    
    return parser.parse_args()


def load_expert(expert_path, expert_algo, env):
    """Load expert policy from file."""
    try:
        if expert_algo.lower() == "ppo":
            expert = PPO.load(expert_path)
        elif expert_algo.lower() == "sac":
            expert = SAC.load(expert_path)
        elif expert_algo.lower() == "td3":
            expert = TD3.load(expert_path)
        else:
            raise ValueError(f"Unsupported algorithm: {expert_algo}")
        
        print(f"Successfully loaded {expert_algo.upper()} expert from {expert_path}")
        return expert
    except Exception as e:
        print(f"Error loading expert: {e}")
        raise


def main(args):
    """Main function to collect expert trajectories."""
    # Create environment
    env = gym.make(args.env)
    
    # Load expert policy
    expert = load_expert(args.expert_path, args.expert_algo, env)
    
    # Collect trajectories
    print(f"Collecting {args.n_episodes} expert episodes from {args.env}...")
    trajectories = collect_expert_trajectories(
        policy=expert,
        env=env,
        n_episodes=args.n_episodes,
        render=args.render,
        deterministic=args.deterministic
    )
    
    # Save trajectories
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    save_expert_trajectories(trajectories, args.output_path)
    
    # Print statistics
    mean_return = trajectories['mean_return']
    std_return = trajectories['std_return']
    total_transitions = len(trajectories['states'])
    
    print(f"Successfully collected {total_transitions} transitions from {args.n_episodes} episodes")
    print(f"Average return: {mean_return:.2f} ± {std_return:.2f}")
    print(f"Expert trajectories saved to {args.output_path}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)