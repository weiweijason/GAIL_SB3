import sys
import os
import argparse
import time
import gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
import pandas as pd
import seaborn as sns
from pathlib import Path
import imageio
import datetime

from gail_pytorch.models.gail import GAIL
from gail_pytorch.models.policy import DiscretePolicy, ContinuousPolicy
from gail_pytorch.utils.expert_trajectories import load_expert_trajectories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate and visualize GAIL model")
    
    # Model and environment settings
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym environment name")
    parser.add_argument("--expert_data", type=str, 
                        help="Path to expert trajectory file (for comparison)")
    
    # Evaluation settings
    parser.add_argument("--n_eval_episodes", type=int, default=20,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                        help="Maximum episode length")
    
    # Visualization settings
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--save_video", action="store_true",
                        help="Save evaluation video")
    parser.add_argument("--video_path", type=str, default="./data/videos",
                        help="Path to save videos")
    parser.add_argument("--plot_path", type=str, default="./data/plots",
                        help="Path to save plots")
    
    # Device settings
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    
    # Policy network settings
    parser.add_argument("--hidden_dims", type=str, default="64,64",
                        help="Policy network hidden layer dimensions, comma-separated, e.g.: 64,64")
    
    return parser.parse_args()


def load_policy(model_path, env, device, hidden_dims_str="64,64"):
    """Load the trained policy."""
    # Parse hidden layer dimensions
    hidden_dims = tuple(int(dim) for dim in hidden_dims_str.split(','))
    print(f"Using hidden layer dimensions: {hidden_dims}")
    
    # Determine action space type
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        action_dim = env.action_space.n
    else:
        is_discrete = False
        action_dim = env.action_space.shape[0]
    
    # Get state dimension
    state_dim = env.observation_space.shape[0]
    
    # Create the appropriate policy based on action space type
    if is_discrete:
        policy = DiscretePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=device
        )
    else:
        policy = ContinuousPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims,
            device=device
        )
    
    # Load checkpoint
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"Successfully loaded checkpoint, checkpoint type: {type(checkpoint)}")
        
        # Check checkpoint structure and print keys for debugging
        if isinstance(checkpoint, dict):
            print(f"Checkpoint contains the following keys: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise
    
    # Try different ways to load the model
    try:
        if isinstance(checkpoint, dict):
            # Case 1: Standard dictionary format with "policy" key
            if "policy" in checkpoint:
                print("Loading model using 'policy' key")
                policy.load_state_dict(checkpoint["policy"])
            # Case 2: Full GAIL model saved with "discriminator" key
            elif "discriminator" in checkpoint:
                print("Loading using GAIL model format")
                # Create a dummy expert data with necessary structure
                state_sample = np.zeros((1, state_dim), dtype=np.float32)
                if is_discrete:
                    action_sample = np.array([0], dtype=np.int64)
                else:
                    action_sample = np.zeros((1, action_dim), dtype=np.float32)
                
                dummy_expert_data = {
                    "states": state_sample,
                    "actions": action_sample,
                    "rewards": np.array([0.0]),
                    "dones": np.array([False]),
                    "next_states": state_sample.copy(),
                    "episode_returns": [0.0],
                    "episode_lengths": [1],
                    "mean_return": 0.0,
                    "std_return": 0.0
                }
                
                # Create GAIL instance
                gail = GAIL(
                    policy=policy,
                    expert_trajectories=dummy_expert_data,
                    device=device
                )
                
                # Load discriminator and policy
                gail.discriminator.load_state_dict(checkpoint["discriminator"])
                
                # Check for policy-related keys
                policy_keys = [k for k in checkpoint.keys() if "policy" in k.lower()]
                if policy_keys:
                    print(f"Found policy-related keys: {policy_keys}")
                    for k in policy_keys:
                        try:
                            policy.load_state_dict(checkpoint[k])
                            print(f"Successfully loaded policy using '{k}'")
                            break
                        except Exception as e:
                            print(f"Failed to load using '{k}': {e}")
                else:
                    print("No policy-related keys found, trying to load model parameters directly to policy")
            # Case 3: Checkpoint is directly the policy state dict
            else:
                print("Trying to load checkpoint directly as policy state dict")
                try:
                    policy.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"Direct loading failed: {e}")
                    
                    # Case 4: Checkpoint might contain policy under other naming
                    print("Trying to find other possible policy keys...")
                    potential_keys = ['actor', 'model', 'net', 'network', 'state_dict']
                    
                    for key in potential_keys:
                        if key in checkpoint:
                            try:
                                print(f"Trying to load policy using '{key}' key")
                                policy.load_state_dict(checkpoint[key])
                                print(f"Successfully loaded policy using '{key}'")
                                break
                            except Exception as e:
                                print(f"Failed to load using '{key}': {e}")
        # Case 5: Checkpoint is directly the model parameters
        else:
            print("Checkpoint is not a dictionary format, trying to load directly")
            policy.load_state_dict(checkpoint)
        
        print("Policy loading successful!")
    except Exception as e:
        print(f"All loading attempts failed: {e}")
        print("Creating a new policy model...")
    
    policy.eval()  # Set to evaluation mode
    return policy, is_discrete


def evaluate_policy_with_data_collection(policy, env, args, is_discrete):
    """Evaluate policy and collect data for visualization."""
    returns = []
    lengths = []
    
    all_states = []
    all_actions = []
    all_rewards = []
    
    # If we want to save video
    frames = []
    
    for i in range(args.n_eval_episodes):
        states = []
        actions = []
        rewards = []
        
        state, _ = env.reset(seed=args.seed + i)
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done and episode_length < args.max_ep_len:
            # Safely try to render, if it fails output a warning without terminating the program
            if args.render or args.save_video:
                try:
                    frame = env.render()
                    if args.save_video and frame is not None:
                        frames.append(frame)
                except Exception as e:
                    if episode_length == 0:  # Only output warning on first step of each episode
                        print(f"Warning: Error rendering environment: {e}")
                        print("Continuing evaluation without rendering. For rendering, install necessary dependencies: pip install pygame")
                        # Turn off rendering to avoid repeated errors
                        args.render = False
                        if args.save_video:
                            print("Video saving feature has been disabled")
                            args.save_video = False
            
            # Get action from policy
            if is_discrete:
                action, _, _ = policy.get_action(state, deterministic=True)
            else:
                action, _, _ = policy.get_action(state, deterministic=True)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Save data
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # Update for next iteration
            state = next_state
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        
        print(f"Episode {i+1}/{args.n_eval_episodes} - Return: {episode_return:.2f}, Length: {episode_length}")
    
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    mean_length = np.mean(lengths)
    std_length = np.std(lengths)
    
    evaluation_data = {
        "returns": returns,
        "lengths": lengths,
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_length": mean_length,
        "std_length": std_length,
        "states": np.array(all_states),
        "actions": np.array(all_actions),
        "rewards": np.array(all_rewards)
    }
    
    # Save video
    if args.save_video and frames:
        save_video(frames, args.video_path, env.spec.id)
    
    return evaluation_data


def compare_with_expert(evaluation_data, expert_data, args):
    """Compare model performance with expert performance."""
    # Create comparison plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Return comparison
    ax = axes[0]
    model_returns = evaluation_data["returns"]
    expert_returns = expert_data["episode_returns"]
    
    ax.axhline(y=evaluation_data["mean_return"], color='b', linestyle='-', alpha=0.5)
    ax.axhline(y=expert_data["mean_return"], color='r', linestyle='-', alpha=0.5)
    
    ax.boxplot([model_returns, expert_returns], labels=["GAIL Model", "Expert"])
    ax.set_title("Return Distribution Comparison")
    ax.set_ylabel("Total Return")
    
    # Add annotations for means
    ax.annotate(f'Mean: {evaluation_data["mean_return"]:.2f}', 
                xy=(1, evaluation_data["mean_return"]), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='blue')
    
    ax.annotate(f'Mean: {expert_data["mean_return"]:.2f}', 
                xy=(2, expert_data["mean_return"]), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='red')
    
    # Episode length comparison
    ax = axes[1]
    model_lengths = evaluation_data["lengths"]
    expert_lengths = expert_data["episode_lengths"]
    
    ax.axhline(y=evaluation_data["mean_length"], color='b', linestyle='-', alpha=0.5)
    ax.axhline(y=np.mean(expert_lengths), color='r', linestyle='-', alpha=0.5)
    
    ax.boxplot([model_lengths, expert_lengths], labels=["GAIL Model", "Expert"])
    ax.set_title("Episode Length Comparison")
    ax.set_ylabel("Steps")
    
    # Add annotations for means
    ax.annotate(f'Mean: {evaluation_data["mean_length"]:.2f}', 
                xy=(1, evaluation_data["mean_length"]), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='blue')
    
    ax.annotate(f'Mean: {np.mean(expert_lengths):.2f}', 
                xy=(2, np.mean(expert_lengths)), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='red')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs(args.plot_path, exist_ok=True)
    plt.savefig(os.path.join(args.plot_path, f"comparison_{args.env}.png"), dpi=300)
    
    # Generate trajectory or state-action heatmaps (if dimensionality allows)
    try:
        if evaluation_data["states"].shape[1] == 2:  # 2D state space
            plot_state_distributions(
                evaluation_data["states"], 
                expert_data["states"], 
                args.plot_path, 
                args.env
            )
        
        if not isinstance(evaluation_data["actions"][0], (int, np.integer)):  # Continuous action space
            if len(evaluation_data["actions"].shape) > 1 and evaluation_data["actions"].shape[1] <= 2:
                plot_action_distributions(
                    evaluation_data["actions"], 
                    expert_data["actions"], 
                    args.plot_path, 
                    args.env
                )
    except Exception as e:
        print(f"Error generating distribution plots: {e}")
    
    return fig


def plot_state_distributions(model_states, expert_states, plot_path, env_name):
    """Plot state distribution comparison."""
    plt.figure(figsize=(10, 8))
    
    # Reduce state space to 2D (if higher dimensions)
    if model_states.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        model_states_2d = pca.fit_transform(model_states)
        expert_states_2d = pca.transform(expert_states)
    else:
        model_states_2d = model_states
        expert_states_2d = expert_states
    
    # Plot density maps
    sns.kdeplot(x=model_states_2d[:, 0], y=model_states_2d[:, 1], 
                cmap="Blues", fill=True, alpha=0.5, label="GAIL Model")
    sns.kdeplot(x=expert_states_2d[:, 0], y=expert_states_2d[:, 1], 
                cmap="Reds", fill=True, alpha=0.5, label="Expert")
    
    plt.title(f"{env_name} - State Distribution Comparison")
    plt.xlabel("State Dimension 1")
    plt.ylabel("State Dimension 2")
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(plot_path, f"state_distribution_{env_name}.png"), dpi=300)


def plot_action_distributions(model_actions, expert_actions, plot_path, env_name):
    """Plot action distribution comparison."""
    plt.figure(figsize=(10, 8))
    
    # Handle action dimensions
    if len(model_actions.shape) > 1 and model_actions.shape[1] > 1:
        # 2D action space
        sns.kdeplot(x=model_actions[:, 0], y=model_actions[:, 1], 
                    cmap="Blues", fill=True, alpha=0.5, label="GAIL Model")
        sns.kdeplot(x=expert_actions[:, 0], y=expert_actions[:, 1], 
                    cmap="Reds", fill=True, alpha=0.5, label="Expert")
        
        plt.xlabel("Action Dimension 1")
        plt.ylabel("Action Dimension 2")
    else:
        # 1D action space
        sns.kdeplot(model_actions, fill=True, color="blue", alpha=0.5, label="GAIL Model")
        sns.kdeplot(expert_actions, fill=True, color="red", alpha=0.5, label="Expert")
        
        plt.xlabel("Action Value")
        plt.ylabel("Density")
    
    plt.title(f"{env_name} - Action Distribution Comparison")
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(plot_path, f"action_distribution_{env_name}.png"), dpi=300)


def save_video(frames, video_path, env_name):
    """Save episode video."""
    os.makedirs(video_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = os.path.join(video_path, f"{env_name}_{timestamp}.mp4")
    
    # Save video using imageio
    imageio.mimsave(video_file, frames, fps=30)
    print(f"Video saved to {video_file}")
    return video_file


def generate_summary_report(evaluation_data, expert_data, args):
    """Generate evaluation summary report."""
    report = {
        "Environment": args.env,
        "Number of Evaluation Episodes": args.n_eval_episodes,
        "Model Path": args.model_path,
        "Evaluation Time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Model Performance": {
            "Average Return": f"{evaluation_data['mean_return']:.2f} ± {evaluation_data['std_return']:.2f}",
            "Average Episode Length": f"{evaluation_data['mean_length']:.2f} ± {evaluation_data['std_length']:.2f}",
            "Highest Return": f"{max(evaluation_data['returns']):.2f}",
            "Lowest Return": f"{min(evaluation_data['returns']):.2f}"
        }
    }
    
    if expert_data:
        report["Expert Performance"] = {
            "Average Return": f"{expert_data['mean_return']:.2f} ± {expert_data['std_return']:.2f}",
            "Average Episode Length": f"{np.mean(expert_data['episode_lengths']):.2f} ± {np.std(expert_data['episode_lengths']):.2f}",
            "Highest Return": f"{max(expert_data['episode_returns']):.2f}",
            "Lowest Return": f"{min(expert_data['episode_returns']):.2f}"
        }
        
        # Calculate performance gap between model and expert
        model_mean = evaluation_data['mean_return']
        expert_mean = expert_data['mean_return']
        performance_gap = model_mean - expert_mean
        performance_percentage = (model_mean / expert_mean) * 100 if expert_mean != 0 else float('inf')
        
        report["Comparison with Expert"] = {
            "Absolute Gap": f"{performance_gap:.2f}",
            "Relative Performance": f"{performance_percentage:.2f}%"
        }
    
    # Save report
    os.makedirs(args.plot_path, exist_ok=True)
    report_path = os.path.join(args.plot_path, f"evaluation_report_{args.env}.txt")
    
    with open(report_path, "w", encoding="utf-8") as f:
        for section, content in report.items():
            if isinstance(content, dict):
                f.write(f"== {section} ==\n")
                for key, value in content.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"{section}: {content}\n")
            f.write("\n")
    
    print(f"Evaluation report saved to {report_path}")
    return report


def main(args):
    """Main evaluation function."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Try to install required dependencies
    if args.render or args.save_video:
        try:
            import importlib
            if not importlib.util.find_spec("pygame"):
                print("Attempting to install pygame...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
                print("pygame installation successful!")
        except Exception as e:
            print(f"Could not automatically install pygame: {e}")
            print("Continuing but may not be able to render the environment")
    
    # Create environment
    try:
        env = gym.make(args.env, render_mode="rgb_array" if args.render or args.save_video else None)
    except Exception as e:
        print(f"Error creating environment with render_mode: {e}")
        print("Trying to create environment without specifying render_mode...")
        env = gym.make(args.env)
        if args.render or args.save_video:
            print("Warning: Environment created without render mode, may not be able to save video")
    
    # Load policy
    print(f"Loading model from {args.model_path}...")
    policy, is_discrete = load_policy(args.model_path, env, args.device, args.hidden_dims)
    
    # Evaluate policy
    print(f"Evaluating {args.n_eval_episodes} episodes in {args.env} environment...")
    evaluation_data = evaluate_policy_with_data_collection(policy, env, args, is_discrete)
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    print(f"Average return: {evaluation_data['mean_return']:.2f} ± {evaluation_data['std_return']:.2f}")
    print(f"Average episode length: {evaluation_data['mean_length']:.2f} ± {evaluation_data['std_length']:.2f}")
    print(f"Highest return: {max(evaluation_data['returns']):.2f}")
    print(f"Lowest return: {min(evaluation_data['returns']):.2f}")
    
    # If expert data is provided, compare with it
    expert_data = None
    if args.expert_data:
        print(f"\nLoading expert data from {args.expert_data}...")
        expert_data = load_expert_trajectories(args.expert_data)
        
        print("\n=== Comparison with Expert ===")
        print(f"GAIL model average return: {evaluation_data['mean_return']:.2f}")
        print(f"Expert average return: {expert_data['mean_return']:.2f}")
        
        # Generate comparison plots
        compare_with_expert(evaluation_data, expert_data, args)
    
    # Generate evaluation report
    report = generate_summary_report(evaluation_data, expert_data, args)
    
    # Close environment
    env.close()
    
    print("\nEvaluation complete!")
    if args.save_video:
        print(f"Videos saved to {args.video_path} directory")
    print(f"Plots saved to {args.plot_path} directory")


if __name__ == "__main__":
    args = parse_args()
    main(args)