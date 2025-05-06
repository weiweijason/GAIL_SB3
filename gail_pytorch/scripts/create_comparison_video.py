# Recommended file path: gail_pytorch/scripts/create_comparison_video.py
"""
Create side-by-side comparison videos of GAIL model and expert behavior.

This script loads a GAIL model and expert policy, and creates a side-by-side video to visually compare their behaviors.
"""
import os
import argparse
import time
import gym
import numpy as np
import torch
from pathlib import Path
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import datetime

from gail_pytorch.models.policy import DiscretePolicy, ContinuousPolicy
from gail_pytorch.utils.expert_trajectories import load_expert_trajectories


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create side-by-side comparison videos of GAIL model and expert behavior")
    
    parser.add_argument("--gail_model_path", type=str, required=True,
                      help="Path to the trained GAIL model")
    parser.add_argument("--expert_policy_path", type=str,
                      help="Path to the expert policy model (if available)")
    parser.add_argument("--expert_data", type=str, required=True,
                      help="Path to expert trajectory data (for evaluation or as an alternative to expert policy)")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                      help="Gym environment name")
    parser.add_argument("--n_episodes", type=int, default=3,
                      help="Number of episodes to compare")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                      help="Maximum episode length")
    parser.add_argument("--seed", type=int, default=0,
                      help="Random seed")
    parser.add_argument("--output_path", type=str, default="./data/videos/comparison",
                      help="Path to save output videos")
    parser.add_argument("--fps", type=int, default=30,
                      help="Frames per second for the video")
    parser.add_argument("--hidden_dims", type=str, default="64,64",
                      help="Hidden layer dimensions of the model, comma separated, e.g.: 64,64")
    parser.add_argument("--width", type=int, default=1280,
                      help="Video width")
    parser.add_argument("--height", type=int, default=480,
                      help="Video height")
    parser.add_argument("--device", type=str, default="cuda",
                      help="Device to run on (cuda or cpu)")
    
    return parser.parse_args()


def load_gail_policy(model_path, env, device, hidden_dims_str="64,64"):
    """Load GAIL policy model."""
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
    
    # Create policy network
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
    
    # Load model parameters
    print(f"Loading model from {model_path}...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if "policy" in checkpoint:
                policy.load_state_dict(checkpoint["policy"])
                print("Successfully loaded GAIL policy model")
            elif isinstance(checkpoint, dict) and len(checkpoint) > 0:
                # Try to find policy parameters
                for key in checkpoint.keys():
                    if "policy" in key.lower():
                        policy.load_state_dict(checkpoint[key])
                        print(f"Successfully loaded GAIL policy model using '{key}' key")
                        break
                else:
                    # Try direct loading
                    try:
                        policy.load_state_dict(checkpoint)
                        print("Successfully loaded GAIL policy model directly")
                    except Exception as e:
                        print(f"Failed to load GAIL policy: {e}")
                        raise
        else:
            # Directly a policy state dict
            policy.load_state_dict(checkpoint)
            print("Successfully loaded GAIL policy model directly")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    policy.eval()  # Set to evaluation mode
    return policy, is_discrete


def run_episodes(env, policy, is_discrete, n_episodes, max_steps, seed=None, model_name="Model"):
    """Run multiple episodes and collect rendering frames."""
    all_frames = []
    all_returns = []
    all_lengths = []
    
    for i in range(n_episodes):
        frames = []
        ep_return = 0
        
        if seed is not None:
            state, _ = env.reset(seed=seed+i)
        else:
            state, _ = env.reset()
            
        done = False
        step = 0
        
        while not done and step < max_steps:
            # Render current state
            frame = env.render()
            frames.append(frame)
            
            # Get action from policy
            if is_discrete:
                action, _, _ = policy.get_action(state, deterministic=True)
            else:
                action, _, _ = policy.get_action(state, deterministic=True)
            
            # Execute action in environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and return
            state = next_state
            ep_return += reward
            step += 1
        
        all_frames.append(frames)
        all_returns.append(ep_return)
        all_lengths.append(step)
        
        print(f"{model_name} Episode {i+1}/{n_episodes} - Return: {ep_return:.2f}, Length: {step}")
    
    avg_return = np.mean(all_returns)
    avg_length = np.mean(all_lengths)
    print(f"{model_name} Average Return: {avg_return:.2f}, Average Length: {avg_length:.2f}")
    
    return all_frames, all_returns, all_lengths


def replay_expert_trajectories(env, expert_data, n_episodes, max_steps, seed=None):
    """Replay expert behaviors from trajectory data."""
    all_frames = []
    all_returns = []
    all_lengths = []
    
    # Extract trajectories from expert data
    # Assume expert data contains consecutive time steps and complete episodes can be reconstructed
    states = expert_data["states"]
    actions = expert_data["actions"]
    rewards = expert_data.get("rewards", [])
    dones = expert_data.get("dones", [])
    episode_returns = expert_data.get("episode_returns", [])
    episode_lengths = expert_data.get("episode_lengths", [])
    
    print("\n=== Replaying Expert Trajectories ===")
    
    # If episode_returns available, use these directly instead of recalculating
    if episode_returns and len(episode_returns) >= n_episodes:
        # Select episodes to replay
        if seed is not None:
            np.random.seed(seed)
        
        if len(episode_returns) <= n_episodes:
            selected_episodes = range(len(episode_returns))
        else:
            selected_episodes = np.random.choice(len(episode_returns), n_episodes, replace=False)
            
        # Get episode boundaries
        episode_ends = [i for i, done in enumerate(dones) if done] if len(dones) > 0 else []
        if not episode_ends:
            # If no explicit episode endings, try to infer from episode_lengths
            if episode_lengths:
                episode_ends = []
                current_end = 0
                for length in episode_lengths:
                    current_end += length
                    episode_ends.append(current_end - 1)  # Convert to 0-indexing
        
        # If still can't determine episode boundaries, assume all data is one episode
        if not episode_ends:
            episode_ends = [len(states) - 1]
        
        # Replay each selected episode
        for idx, ep_idx in enumerate(selected_episodes):
            frames = []
            start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1] + 1
            end_idx = episode_ends[ep_idx]
            
            # Set environment to initial state
            state, _ = env.reset()
            
            # Replay this episode
            ep_return = episode_returns[ep_idx]  # Use stored return instead of recalculating
            episode_length = min(end_idx - start_idx + 1, max_steps)
            
            for i in range(episode_length):
                idx = start_idx + i
                if idx > end_idx:
                    break
                    
                # Render current state
                frame = env.render()
                frames.append(frame)
                
                # Get expert action and execute (but don't accumulate rewards)
                action = actions[idx]
                next_state, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update state
                state = next_state
                
                if done:
                    break
            
            all_frames.append(frames)
            all_returns.append(ep_return)
            all_lengths.append(episode_length)
            
            print(f"Expert Episode {ep_idx+1} - Return: {ep_return:.2f}, Length: {episode_length}")
    
    # Fallback to original method if episode_returns not available
    else:
        # Get episode boundaries
        episode_ends = [i for i, done in enumerate(dones) if done] if len(dones) > 0 else []
        if not episode_ends:
            # If no explicit episode endings, try to infer from other information
            if "episode_lengths" in expert_data:
                lengths = expert_data["episode_lengths"]
                episode_ends = []
                current_end = 0
                for length in lengths:
                    current_end += length
                    episode_ends.append(current_end - 1)  # Convert to 0-indexing
        
        # If still can't determine episode boundaries, assume all data is one episode
        if not episode_ends:
            episode_ends = [len(states) - 1]
        
        # Select episodes to replay
        if seed is not None:
            np.random.seed(seed)
        
        if len(episode_ends) <= n_episodes:
            selected_episodes = range(len(episode_ends))
        else:
            selected_episodes = np.random.choice(len(episode_ends), n_episodes, replace=False)
        
        # Replay each selected episode
        for ep_idx in selected_episodes:
            frames = []
            start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1] + 1
            end_idx = episode_ends[ep_idx]
            
            # Set environment to initial state
            state, _ = env.reset()
            
            # Replay this episode
            ep_return = 0
            episode_length = min(end_idx - start_idx + 1, max_steps)
            
            for i in range(episode_length):
                idx = start_idx + i
                if idx > end_idx:
                    break
                    
                # Render current state
                frame = env.render()
                frames.append(frame)
                
                # Get expert action and execute
                action = actions[idx]
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                # Update state and return
                state = next_state
                ep_return += reward
                
                if done:
                    break
            
            all_frames.append(frames)
            all_returns.append(ep_return)
            all_lengths.append(episode_length)
            
            print(f"Expert Episode {ep_idx+1} - Return: {ep_return:.2f}, Length: {episode_length}")
    
    avg_return = np.mean(all_returns)
    avg_length = np.mean(all_lengths)
    print(f"Expert Average Return: {avg_return:.2f}, Average Length: {avg_length:.2f}")
    
    return all_frames, all_returns, all_lengths


def load_expert_policy(policy_path, env, device, hidden_dims_str="64,64"):
    """Load expert policy model (if available)."""
    # Similar to load_gail_policy function
    return load_gail_policy(policy_path, env, device, hidden_dims_str)


def create_side_by_side_frame(gail_frame, expert_frame, gail_info, expert_info, width=1280, height=480):
    """Create side-by-side comparison frame."""
    # Create a new figure
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    canvas = FigureCanvas(fig)
    
    # Ensure both frames have the same size
    if gail_frame.shape != expert_frame.shape:
        # Resize to make dimensions consistent
        from skimage.transform import resize
        min_height = min(gail_frame.shape[0], expert_frame.shape[0])
        min_width = min(gail_frame.shape[1], expert_frame.shape[1])
        gail_frame = resize(gail_frame, (min_height, min_width), preserve_range=True).astype(np.uint8)
        expert_frame = resize(expert_frame, (min_height, min_width), preserve_range=True).astype(np.uint8)
    
    # Create side-by-side layout
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(gail_frame)
    ax1.set_title(f"GAIL Model\n{gail_info}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(expert_frame)
    ax2.set_title(f"Expert Behavior\n{expert_info}")
    ax2.axis('off')
    
    fig.tight_layout()
    
    # Convert to image
    canvas.draw()
    comparison_frame = np.array(canvas.renderer.buffer_rgba())
    
    plt.close(fig)
    return comparison_frame


def create_comparison_video(gail_frames_list, expert_frames_list, gail_returns, expert_returns, 
                           gail_lengths, expert_lengths, output_path, fps=30, width=1280, height=480):
    """Create side-by-side comparison video of GAIL model and expert behavior."""
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f"comparison_{timestamp}.mp4")
    
    # Ensure we have the same number of episodes
    n_episodes = min(len(gail_frames_list), len(expert_frames_list))
    
    # Create a video for each episode
    for ep in range(n_episodes):
        gail_frames = gail_frames_list[ep]
        expert_frames = expert_frames_list[ep]
        
        # Ensure we have enough frames to compare
        max_frames = max(len(gail_frames), len(expert_frames))
        
        # Prepare comparison frames
        comparison_frames = []
        
        # Create info strings
        gail_info = f"Return: {gail_returns[ep]:.2f}, Length: {gail_lengths[ep]}"
        expert_info = f"Return: {expert_returns[ep]:.2f}, Length: {expert_lengths[ep]}"
        
        for i in range(max_frames):
            # Get GAIL frame (if available, otherwise use last frame)
            gail_idx = min(i, len(gail_frames) - 1)
            gail_frame = gail_frames[gail_idx]
            
            # Get expert frame (if available, otherwise use last frame)
            expert_idx = min(i, len(expert_frames) - 1)
            expert_frame = expert_frames[expert_idx]
            
            # Create side-by-side comparison
            comparison = create_side_by_side_frame(
                gail_frame, expert_frame, gail_info, expert_info, width, height
            )
            comparison_frames.append(comparison)
        
        # Save this episode's video
        ep_output_file = os.path.join(output_path, f"comparison_ep{ep+1}_{timestamp}.mp4")
        imageio.mimsave(ep_output_file, comparison_frames, fps=fps)
        print(f"Episode {ep+1} comparison video saved to {ep_output_file}")
    
    # Create a merged video of all episodes
    all_comparison_frames = []
    for ep in range(n_episodes):
        gail_frames = gail_frames_list[ep]
        expert_frames = expert_frames_list[ep]
        
        # Ensure we have enough frames to compare
        max_frames = max(len(gail_frames), len(expert_frames))
        
        # Create info strings
        gail_info = f"Episode {ep+1}: Return={gail_returns[ep]:.2f}, Length={gail_lengths[ep]}"
        expert_info = f"Episode {ep+1}: Return={expert_returns[ep]:.2f}, Length={expert_lengths[ep]}"
        
        for i in range(max_frames):
            # Get GAIL frame (if available, otherwise use last frame)
            gail_idx = min(i, len(gail_frames) - 1)
            gail_frame = gail_frames[gail_idx]
            
            # Get expert frame (if available, otherwise use last frame)
            expert_idx = min(i, len(expert_frames) - 1)
            expert_frame = expert_frames[expert_idx]
            
            # Create side-by-side comparison
            comparison = create_side_by_side_frame(
                gail_frame, expert_frame, gail_info, expert_info, width, height
            )
            all_comparison_frames.append(comparison)
        
        # Add a brief black frame as a separator between episodes
        if ep < n_episodes - 1:
            separator = np.zeros((height, width, 4), dtype=np.uint8)
            for _ in range(int(fps/2)):  # 0.5 seconds of separation
                all_comparison_frames.append(separator)
    
    # Save the merged video
    imageio.mimsave(output_file, all_comparison_frames, fps=fps)
    print(f"All episodes comparison video saved to {output_file}")
    
    return output_file


def main(args):
    """Main function."""
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create environment (with rendering option)
    try:
        env = gym.make(args.env, render_mode="rgb_array")
    except Exception as e:
        print(f"Error creating environment with render mode: {e}")
        print("Trying to create environment without specifying render_mode...")
        env = gym.make(args.env)
    
    # Load GAIL policy
    gail_policy, is_discrete = load_gail_policy(
        args.gail_model_path, env, args.device, args.hidden_dims
    )
    
    # Run GAIL model and collect frames
    print("\n=== Running GAIL Model ===")
    gail_frames, gail_returns, gail_lengths = run_episodes(
        env, gail_policy, is_discrete, args.n_episodes, args.max_ep_len, args.seed, "GAIL"
    )
    
    # Load expert data
    expert_data = load_expert_trajectories(args.expert_data)
    print(f"\nLoaded expert data with {len(expert_data['states'])} time steps")
    
    # If expert policy path is provided, load expert policy
    if args.expert_policy_path and os.path.exists(args.expert_policy_path):
        print("\n=== Loading Expert Policy ===")
        expert_policy, expert_is_discrete = load_expert_policy(
            args.expert_policy_path, env, args.device, args.hidden_dims
        )
        
        print("\n=== Running Expert Policy ===")
        expert_frames, expert_returns, expert_lengths = run_episodes(
            env, expert_policy, expert_is_discrete, args.n_episodes, 
            args.max_ep_len, args.seed + 100, "Expert"
        )
    else:
        # Use expert trajectory data to replay expert behavior
        print("\n=== Replaying Expert Trajectories ===")
        expert_frames, expert_returns, expert_lengths = replay_expert_trajectories(
            env, expert_data, args.n_episodes, args.max_ep_len, args.seed + 100
        )
    
    # Create comparison video
    print("\n=== Creating Comparison Video ===")
    output_file = create_comparison_video(
        gail_frames, expert_frames, gail_returns, expert_returns,
        gail_lengths, expert_lengths, args.output_path, args.fps,
        args.width, args.height
    )
    
    print(f"\nComparison video successfully created: {output_file}")
    print(f"GAIL average return: {np.mean(gail_returns):.2f} ± {np.std(gail_returns):.2f}")
    print(f"Expert average return: {np.mean(expert_returns):.2f} ± {np.std(expert_returns):.2f}")
    
    # Close environment
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)