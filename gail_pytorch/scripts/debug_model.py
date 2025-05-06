# Recommended file path: gail_pytorch/scripts/debug_model.py
"""
Debug script for GAIL model loading and evaluation.

This script is used for detailed diagnosis of issues in model loading and evaluation process.
"""
import os
import sys
import argparse
import numpy as np
import torch
import gym

from gail_pytorch.models.policy import DiscretePolicy, ContinuousPolicy

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Debug GAIL model")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym environment name")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                        help="Maximum episode length")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run on (cuda or cpu)")
    
    return parser.parse_args()

def debug_model_structure(model_path, device):
    """Check model file structure and print detailed information."""
    print(f"\n=== Examining Model File Structure ===")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"Model type: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"Keys in the model: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    print(f"  - {key}: Dictionary with {len(value)} items")
                elif isinstance(value, torch.Tensor):
                    print(f"  - {key}: Tensor with shape {value.shape}")
                else:
                    print(f"  - {key}: {type(value)}")
        else:
            print("Model is not in dictionary format, cannot view internal structure")
            
    except Exception as e:
        print(f"Error while checking model structure: {e}")

def run_single_episode(policy, env, max_steps=1000, seed=None, debug=True):
    """Run a single episode and print detailed debug information."""
    returns = 0
    length = 0
    
    # Reset environment
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()
    
    done = False
    
    if debug:
        print("\n=== Episode Details ===")
    
    while not done and length < max_steps:
        # Get action from policy
        if isinstance(policy, DiscretePolicy):
            action_logits, _ = policy(torch.FloatTensor(state).unsqueeze(0).to(policy.device))
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            
            if debug and length == 0:
                print(f"Action probabilities: {action_probs.detach().cpu().numpy()}")
            
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print(f"Warning: Action probabilities contain NaN or Inf values!")
                
            action, log_prob, _ = policy.get_action(state, deterministic=True)
            
            if debug and length < 5:  # Only print first 5 steps to avoid too much information
                print(f"Step {length}: State {state}, Action {action}, Action probabilities {action_probs.detach().cpu().numpy()}")
        else:
            action_means, action_log_stds, _ = policy(torch.FloatTensor(state).unsqueeze(0).to(policy.device))
            
            if debug and length == 0:
                print(f"Action means: {action_means.detach().cpu().numpy()}, Action log stds: {action_log_stds.detach().cpu().numpy()}")
            
            if torch.isnan(action_means).any() or torch.isinf(action_means).any():
                print(f"Warning: Action means contain NaN or Inf values!")
                
            action, log_prob, _ = policy.get_action(state, deterministic=True)
            
            if debug and length < 5:
                print(f"Step {length}: State {state}, Action {action}")
        
        # Execute action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update values
        state = next_state
        returns += reward
        length += 1
    
    if debug:
        print(f"Episode end: Total return {returns}, Length {length}, Natural termination: {done}")
    
    return returns, length, done

def main(args):
    """Main debug function."""
    print("\n=== GAIL Model Debugging Tool ===")
    print(f"Model path: {args.model_path}")
    print(f"Environment: {args.env}")
    print(f"Device: {args.device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Detailed model structure check
    debug_model_structure(args.model_path, args.device)
    
    # Create environment
    env = gym.make(args.env)
    
    # Determine action space and state space
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        action_dim = env.action_space.n
        print(f"\nDiscrete action space: {action_dim} possible actions")
    else:
        is_discrete = False
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
        print(f"\nContinuous action space: Dimension {action_dim}, Range {action_low} to {action_high}")
    
    state_dim = env.observation_space.shape[0]
    print(f"State space: Dimension {state_dim}")
    
    # Create policy network
    if is_discrete:
        policy = DiscretePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(256, 256),
            device=args.device
        )
    else:
        policy = ContinuousPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(256, 256),
            device=args.device
        )
    
    # Load model
    try:
        print("\nAttempting to load model...")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        loaded = False
        
        if isinstance(checkpoint, dict):
            # Regular loading attempts
            potential_keys = ['policy', 'model', 'network', 'actor', 'state_dict']
            for key in potential_keys:
                if key in checkpoint:
                    try:
                        policy.load_state_dict(checkpoint[key])
                        print(f"Successfully loaded model using '{key}' key")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"Failed to load using '{key}': {e}")
            
            # Special handling for GAIL - try to find separate policy file
            if not loaded and 'discriminator' in checkpoint:
                print("\nDetected GAIL model format, but missing policy network parameters")
                print("Attempting to find other possible policy files...")
                
                # Try to load differently named policy files from the same directory
                model_dir = os.path.dirname(args.model_path)
                base_name = os.path.basename(args.model_path).replace('gail_model', '').replace('.pt', '')
                
                potential_policy_files = [
                    os.path.join(model_dir, f"policy_model{base_name}.pt"),
                    os.path.join(model_dir, f"policy{base_name}.pt"),
                    os.path.join(model_dir, f"actor{base_name}.pt"),
                    # Check earlier checkpoints
                    os.path.join(model_dir, "gail_model_*00000.pt"),
                    os.path.join(model_dir, "policy_*.pt")
                ]
                
                # List all files in the directory
                try:
                    all_files = os.listdir(model_dir)
                    print(f"Files in directory {model_dir}: {all_files}")
                    
                    # Check if there are any potential policy files
                    policy_candidates = [f for f in all_files if "policy" in f.lower() or "actor" in f.lower() or ("model" in f.lower() and "gail" not in f.lower())]
                    
                    if policy_candidates:
                        print(f"Found potential policy files: {policy_candidates}")
                        for policy_file in policy_candidates:
                            try:
                                full_path = os.path.join(model_dir, policy_file)
                                print(f"Attempting to load {full_path}...")
                                policy_checkpoint = torch.load(full_path, map_location=args.device)
                                
                                if isinstance(policy_checkpoint, dict):
                                    for key in potential_keys:
                                        if key in policy_checkpoint:
                                            try:
                                                policy.load_state_dict(policy_checkpoint[key])
                                                print(f"Successfully loaded policy from {policy_file} using '{key}' key")
                                                loaded = True
                                                break
                                            except Exception as e:
                                                print(f"Failed to load from {policy_file} using '{key}': {e}")
                                    
                                    if not loaded:
                                        try:
                                            policy.load_state_dict(policy_checkpoint)
                                            print(f"Successfully loaded policy directly from {policy_file}")
                                            loaded = True
                                            break
                                        except Exception as e:
                                            print(f"Failed to load directly from {policy_file}: {e}")
                                else:
                                    try:
                                        policy.load_state_dict(policy_checkpoint)
                                        print(f"Successfully loaded policy from {policy_file}")
                                        loaded = True
                                        break
                                    except Exception as e:
                                        print(f"Failed to load from {policy_file}: {e}")
                            except Exception as e:
                                print(f"Error while processing {policy_file}: {e}")
                
                except Exception as e:
                    print(f"Error listing directory contents: {e}")
                
                # Create new policy model path in the format of GAIL training script
                if not loaded:
                    # Look for intermediate checkpoints saved during training
                    import glob
                    checkpoint_pattern = os.path.join(model_dir, "gail_model_*.pt")
                    checkpoints = glob.glob(checkpoint_pattern)
                    
                    if checkpoints:
                        # Sort by modification time, prioritize newer checkpoints
                        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        print(f"Found {len(checkpoints)} checkpoints, trying to load sequentially...")
                        
                        for checkpoint_path in checkpoints[:5]:  # Only try the 5 most recent checkpoints
                            if checkpoint_path != args.model_path:  # Avoid trying the current file again
                                try:
                                    print(f"Attempting to load policy from checkpoint {checkpoint_path}...")
                                    alt_checkpoint = torch.load(checkpoint_path, map_location=args.device)
                                    
                                    if isinstance(alt_checkpoint, dict) and 'policy' in alt_checkpoint:
                                        try:
                                            policy.load_state_dict(alt_checkpoint['policy'])
                                            print(f"Successfully loaded policy from checkpoint {checkpoint_path}")
                                            loaded = True
                                            break
                                        except Exception as e:
                                            print(f"Failed to load from checkpoint {checkpoint_path}: {e}")
                                except Exception as e:
                                    print(f"Error processing checkpoint {checkpoint_path}: {e}")
            
            # If all attempts fail, create a random policy and provide warning
            if not loaded:
                if 'discriminator' in checkpoint:
                    print("\nWarning: Could not find a valid policy model. Detected GAIL model with only discriminator but no policy parameters.")
                    print("This suggests the model might not have been saved correctly, or policy parameters are stored in a separate file.")
                    print("Consider checking the model saving logic in the training script to ensure complete policy parameter saving.")
                    
                    # As a temporary solution, try to use discriminator parameters directly
                    try:
                        disc_params = checkpoint['discriminator']
                        print("Attempting to initialize policy directly from discriminator parameters...")
                        
                        # Count discriminator parameters and print
                        disc_size = sum(p.numel() for p in disc_params.values())
                        print(f"Total discriminator parameters: {disc_size}")
                        
                        # Print discriminator layer structure
                        for name, param in disc_params.items():
                            if isinstance(param, torch.Tensor):
                                print(f"  - {name}: {param.shape}")
                    except Exception as e:
                        print(f"Error analyzing discriminator parameters: {e}")
                else:
                    print("\nWarning: Could not load any policy model. Will use randomly initialized policy for evaluation.")
                print("Model evaluation results may be inaccurate, reflecting a random policy rather than a trained one.")
        else:
            try:
                policy.load_state_dict(checkpoint)
                print("Successfully loaded model directly")
                loaded = True
            except Exception as e:
                print(f"Direct loading failed: {e}")
                print("\nWarning: Could not load model, will use randomly initialized policy for evaluation.")
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set to evaluation mode
    policy.eval()
    
    # Run a single episode and collect debug information
    print("\nRunning a single episode for detailed debugging...")
    returns, length, done = run_single_episode(policy, env, max_steps=args.max_ep_len, seed=args.seed, debug=True)
    
    print("\n=== Episode Summary ===")
    print(f"Total return: {returns}")
    print(f"Episode length: {length}")
    print(f"Natural termination: {done}")
    
    # Run multiple episodes to calculate average performance
    n_eval_episodes = 5
    all_returns = []
    all_lengths = []
    
    print(f"\nRunning {n_eval_episodes} episodes for evaluation...")
    
    for i in range(n_eval_episodes):
        returns, length, _ = run_single_episode(policy, env, max_steps=args.max_ep_len, seed=args.seed+i, debug=False)
        all_returns.append(returns)
        all_lengths.append(length)
        print(f"Episode {i+1}/{n_eval_episodes} - Return: {returns:.2f}, Length: {length}")
    
    print("\n=== Evaluation Results ===")
    print(f"Average return: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"Average episode length: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
    print(f"Highest return: {max(all_returns):.2f}")
    print(f"Lowest return: {min(all_returns):.2f}")
    
    # Close environment
    env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)