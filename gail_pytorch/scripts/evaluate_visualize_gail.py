# 建議檔案路徑：gail_pytorch/scripts/evaluate_visualize_gail.py
"""
評估和可視化GAIL模型的腳本。

此腳本用於加載訓練好的GAIL模型，進行評估並生成各種可視化結果，
以便展示模型的表現。
"""
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
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description="評估和可視化GAIL模型")
    
    # 模型和環境設置
    parser.add_argument("--model_path", type=str, required=True,
                        help="已訓練模型的路徑")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym環境名稱")
    parser.add_argument("--expert_data", type=str, 
                        help="專家軌跡檔案路徑（用於比較）")
    
    # 評估設置
    parser.add_argument("--n_eval_episodes", type=int, default=20,
                        help="評估的回合數")
    parser.add_argument("--seed", type=int, default=0,
                        help="隨機種子")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                        help="最大回合長度")
    
    # 可視化設置
    parser.add_argument("--render", action="store_true",
                        help="渲染環境")
    parser.add_argument("--save_video", action="store_true",
                        help="保存評估視頻")
    parser.add_argument("--video_path", type=str, default="./data/videos",
                        help="視頻保存路徑")
    parser.add_argument("--plot_path", type=str, default="./data/plots",
                        help="圖表保存路徑")
    
    # 設備設置
    parser.add_argument("--device", type=str, default="cuda",
                        help="運行設備 (cuda 或 cpu)")
    
    return parser.parse_args()


def load_policy(model_path, env, device):
    """加載已訓練的策略。"""
    # 確定動作空間類型
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        action_dim = env.action_space.n
    else:
        is_discrete = False
        action_dim = env.action_space.shape[0]
    
    # 獲取狀態維度
    state_dim = env.observation_space.shape[0]
    
    # 根據動作空間類型創建相應的策略
    if is_discrete:
        policy = DiscretePolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(256, 256),
            device=device
        )
    else:
        policy = ContinuousPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=(256, 256),
            device=device
        )
    
    # 加載檢查點
    checkpoint = torch.load(model_path, map_location=device)
    
    # 判斷是GAIL模型還是單純的策略模型
    if "policy" in checkpoint:
        policy.load_state_dict(checkpoint["policy"])
    else:
        # 假設是通過GAIL類保存的模型
        # 此處需要創建臨時數據以初始化GAIL實例
        dummy_expert_data = {"states": [], "actions": [], "mean_return": 0}
        gail = GAIL(
            policy=policy,
            expert_trajectories=dummy_expert_data,
            device=device
        )
        # 加載GAIL模型
        gail.policy.load_state_dict(checkpoint["policy"])
    
    policy.eval()  # 設置為評估模式
    
    return policy, is_discrete


def evaluate_policy_with_data_collection(policy, env, args, is_discrete):
    """評估策略並收集數據用於可視化。"""
    returns = []
    lengths = []
    
    all_states = []
    all_actions = []
    all_rewards = []
    
    # 如果要保存視頻
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
            if args.render or args.save_video:
                frame = env.render()
                if args.save_video:
                    frames.append(frame)
            
            # 從策略獲取動作
            if is_discrete:
                action, _, _ = policy.get_action(state, deterministic=True)
            else:
                action, _, _ = policy.get_action(state, deterministic=True)
            
            # 在環境中執行動作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 保存數據
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            
            # 為下一次迭代更新
            state = next_state
            episode_return += reward
            episode_length += 1
        
        returns.append(episode_return)
        lengths.append(episode_length)
        
        all_states.extend(states)
        all_actions.extend(actions)
        all_rewards.extend(rewards)
        
        print(f"回合 {i+1}/{args.n_eval_episodes} - 回報: {episode_return:.2f}, 長度: {episode_length}")
    
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
    
    # 保存視頻
    if args.save_video and frames:
        save_video(frames, args.video_path, env.spec.id)
    
    return evaluation_data


def compare_with_expert(evaluation_data, expert_data, args):
    """比較模型與專家表現。"""
    # 創建比較圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 回報比較
    ax = axes[0]
    model_returns = evaluation_data["returns"]
    expert_returns = expert_data["episode_returns"]
    
    ax.axhline(y=evaluation_data["mean_return"], color='b', linestyle='-', alpha=0.5)
    ax.axhline(y=expert_data["mean_return"], color='r', linestyle='-', alpha=0.5)
    
    ax.boxplot([model_returns, expert_returns], labels=["GAIL模型", "專家"])
    ax.set_title("回報分布比較")
    ax.set_ylabel("總回報")
    
    # 為平均值添加註解
    ax.annotate(f'平均: {evaluation_data["mean_return"]:.2f}', 
                xy=(1, evaluation_data["mean_return"]), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='blue')
    
    ax.annotate(f'平均: {expert_data["mean_return"]:.2f}', 
                xy=(2, expert_data["mean_return"]), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='red')
    
    # 回合長度比較
    ax = axes[1]
    model_lengths = evaluation_data["lengths"]
    expert_lengths = expert_data["episode_lengths"]
    
    ax.axhline(y=evaluation_data["mean_length"], color='b', linestyle='-', alpha=0.5)
    ax.axhline(y=np.mean(expert_lengths), color='r', linestyle='-', alpha=0.5)
    
    ax.boxplot([model_lengths, expert_lengths], labels=["GAIL模型", "專家"])
    ax.set_title("回合長度比較")
    ax.set_ylabel("步數")
    
    # 為平均值添加註解
    ax.annotate(f'平均: {evaluation_data["mean_length"]:.2f}', 
                xy=(1, evaluation_data["mean_length"]), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='blue')
    
    ax.annotate(f'平均: {np.mean(expert_lengths):.2f}', 
                xy=(2, np.mean(expert_lengths)), 
                xycoords=('data', 'data'),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom',
                color='red')
    
    plt.tight_layout()
    
    # 保存圖表
    os.makedirs(args.plot_path, exist_ok=True)
    plt.savefig(os.path.join(args.plot_path, f"comparison_{args.env}.png"), dpi=300)
    
    # 生成軌跡或狀態-動作熱圖（如果維度允許）
    try:
        if evaluation_data["states"].shape[1] == 2:  # 2D狀態空間
            plot_state_distributions(
                evaluation_data["states"], 
                expert_data["states"], 
                args.plot_path, 
                args.env
            )
        
        if not isinstance(evaluation_data["actions"][0], (int, np.integer)):  # 連續動作空間
            if len(evaluation_data["actions"].shape) > 1 and evaluation_data["actions"].shape[1] <= 2:
                plot_action_distributions(
                    evaluation_data["actions"], 
                    expert_data["actions"], 
                    args.plot_path, 
                    args.env
                )
    except Exception as e:
        print(f"生成分布圖時出錯: {e}")
    
    return fig


def plot_state_distributions(model_states, expert_states, plot_path, env_name):
    """繪製狀態分布比較。"""
    plt.figure(figsize=(10, 8))
    
    # 將狀態空間縮減為2D（如果維度更高）
    if model_states.shape[1] > 2:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        model_states_2d = pca.fit_transform(model_states)
        expert_states_2d = pca.transform(expert_states)
    else:
        model_states_2d = model_states
        expert_states_2d = expert_states
    
    # 繪製密度圖
    sns.kdeplot(x=model_states_2d[:, 0], y=model_states_2d[:, 1], 
                cmap="Blues", fill=True, alpha=0.5, label="GAIL模型")
    sns.kdeplot(x=expert_states_2d[:, 0], y=expert_states_2d[:, 1], 
                cmap="Reds", fill=True, alpha=0.5, label="專家")
    
    plt.title(f"{env_name} - 狀態分布比較")
    plt.xlabel("狀態維度 1")
    plt.ylabel("狀態維度 2")
    plt.legend()
    
    # 保存圖表
    plt.savefig(os.path.join(plot_path, f"state_distribution_{env_name}.png"), dpi=300)


def plot_action_distributions(model_actions, expert_actions, plot_path, env_name):
    """繪製動作分布比較。"""
    plt.figure(figsize=(10, 8))
    
    # 處理動作維度
    if len(model_actions.shape) > 1 and model_actions.shape[1] > 1:
        # 2D動作空間
        sns.kdeplot(x=model_actions[:, 0], y=model_actions[:, 1], 
                    cmap="Blues", fill=True, alpha=0.5, label="GAIL模型")
        sns.kdeplot(x=expert_actions[:, 0], y=expert_actions[:, 1], 
                    cmap="Reds", fill=True, alpha=0.5, label="專家")
        
        plt.xlabel("動作維度 1")
        plt.ylabel("動作維度 2")
    else:
        # 1D動作空間
        sns.kdeplot(model_actions, fill=True, color="blue", alpha=0.5, label="GAIL模型")
        sns.kdeplot(expert_actions, fill=True, color="red", alpha=0.5, label="專家")
        
        plt.xlabel("動作值")
        plt.ylabel("密度")
    
    plt.title(f"{env_name} - 動作分布比較")
    plt.legend()
    
    # 保存圖表
    plt.savefig(os.path.join(plot_path, f"action_distribution_{env_name}.png"), dpi=300)


def save_video(frames, video_path, env_name):
    """保存回合視頻。"""
    os.makedirs(video_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    video_file = os.path.join(video_path, f"{env_name}_{timestamp}.mp4")
    
    # 使用imageio保存視頻
    imageio.mimsave(video_file, frames, fps=30)
    print(f"視頻已保存到 {video_file}")
    return video_file


def generate_summary_report(evaluation_data, expert_data, args):
    """生成評估摘要報告。"""
    report = {
        "環境": args.env,
        "評估回合數": args.n_eval_episodes,
        "模型路徑": args.model_path,
        "評估時間": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "模型表現": {
            "平均回報": f"{evaluation_data['mean_return']:.2f} ± {evaluation_data['std_return']:.2f}",
            "平均回合長度": f"{evaluation_data['mean_length']:.2f} ± {evaluation_data['std_length']:.2f}",
            "最高回報": f"{max(evaluation_data['returns']):.2f}",
            "最低回報": f"{min(evaluation_data['returns']):.2f}"
        }
    }
    
    if expert_data:
        report["專家表現"] = {
            "平均回報": f"{expert_data['mean_return']:.2f} ± {expert_data['std_return']:.2f}",
            "平均回合長度": f"{np.mean(expert_data['episode_lengths']):.2f} ± {np.std(expert_data['episode_lengths']):.2f}",
            "最高回報": f"{max(expert_data['episode_returns']):.2f}",
            "最低回報": f"{min(expert_data['episode_returns']):.2f}"
        }
        
        # 計算模型與專家的表現差距
        model_mean = evaluation_data['mean_return']
        expert_mean = expert_data['mean_return']
        performance_gap = model_mean - expert_mean
        performance_percentage = (model_mean / expert_mean) * 100 if expert_mean != 0 else float('inf')
        
        report["與專家對比"] = {
            "絕對差距": f"{performance_gap:.2f}",
            "相對表現": f"{performance_percentage:.2f}%"
        }
    
    # 保存報告
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
    
    print(f"評估報告已保存到 {report_path}")
    return report


def main(args):
    """主評估函數。"""
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 創建環境
    env = gym.make(args.env, render_mode="rgb_array" if args.render or args.save_video else None)
    
    # 加載策略
    print(f"從 {args.model_path} 加載模型...")
    policy, is_discrete = load_policy(args.model_path, env, args.device)
    
    # 評估策略
    print(f"在 {args.env} 環境中評估 {args.n_eval_episodes} 個回合...")
    evaluation_data = evaluate_policy_with_data_collection(policy, env, args, is_discrete)
    
    # 打印評估結果
    print("\n=== 評估結果 ===")
    print(f"平均回報: {evaluation_data['mean_return']:.2f} ± {evaluation_data['std_return']:.2f}")
    print(f"平均回合長度: {evaluation_data['mean_length']:.2f} ± {evaluation_data['std_length']:.2f}")
    print(f"最高回報: {max(evaluation_data['returns']):.2f}")
    print(f"最低回報: {min(evaluation_data['returns']):.2f}")
    
    # 如果提供了專家數據，則進行比較
    expert_data = None
    if args.expert_data:
        print(f"\n從 {args.expert_data} 加載專家數據...")
        expert_data = load_expert_trajectories(args.expert_data)
        
        print("\n=== 與專家對比 ===")
        print(f"GAIL模型平均回報: {evaluation_data['mean_return']:.2f}")
        print(f"專家平均回報: {expert_data['mean_return']:.2f}")
        
        # 生成比較圖
        compare_with_expert(evaluation_data, expert_data, args)
    
    # 生成評估報告
    report = generate_summary_report(evaluation_data, expert_data, args)
    
    # 關閉環境
    env.close()
    
    print("\n評估完成！")
    if args.save_video:
        print(f"視頻已保存到 {args.video_path} 目錄")
    print(f"圖表已保存到 {args.plot_path} 目錄")


if __name__ == "__main__":
    args = parse_args()
    main(args)