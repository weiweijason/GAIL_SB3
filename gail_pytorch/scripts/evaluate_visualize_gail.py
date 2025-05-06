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
    print(f"正在從 {model_path} 加載模型...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        print(f"成功加載檢查點，檢查點類型: {type(checkpoint)}")
        
        # 檢查檢查點結構並打印鍵以便調試
        if isinstance(checkpoint, dict):
            print(f"檢查點包含以下鍵: {list(checkpoint.keys())}")
    except Exception as e:
        print(f"加載檢查點出錯: {e}")
        raise
    
    # 嘗試不同的方式加載模型
    try:
        if isinstance(checkpoint, dict):
            # 情況1: 標準字典格式，包含 "policy" 鍵
            if "policy" in checkpoint:
                print("使用 'policy' 鍵加載模型")
                policy.load_state_dict(checkpoint["policy"])
            # 情況2: GAIL 保存的完整模型，包含 "discriminator" 鍵
            elif "discriminator" in checkpoint:
                print("使用 GAIL 模型格式加載")
                # 創建一個包含必要結構的模擬專家數據
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
                
                # 創建 GAIL 實例
                gail = GAIL(
                    policy=policy,
                    expert_trajectories=dummy_expert_data,
                    device=device
                )
                
                # 加載判別器和策略
                gail.discriminator.load_state_dict(checkpoint["discriminator"])
                
                # 檢查策略相關鍵
                policy_keys = [k for k in checkpoint.keys() if "policy" in k.lower()]
                if policy_keys:
                    print(f"找到策略相關鍵: {policy_keys}")
                    for k in policy_keys:
                        try:
                            policy.load_state_dict(checkpoint[k])
                            print(f"使用 '{k}' 成功加載策略")
                            break
                        except:
                            continue
                else:
                    print("未找到策略相關鍵，嘗試直接將模型參數加載到策略中")
            # 情況3: 直接是策略的狀態字典
            else:
                print("嘗試直接將檢查點加載為策略狀態字典")
                try:
                    policy.load_state_dict(checkpoint)
                except Exception as e:
                    print(f"直接加載失敗: {e}")
                    
                    # 情況4: 檢查點可能包含其他命名方式的策略
                    print("嘗試查找其他可能的策略鍵...")
                    potential_keys = ['actor', 'model', 'net', 'network', 'state_dict']
                    
                    for key in potential_keys:
                        if key in checkpoint:
                            try:
                                print(f"嘗試使用 '{key}' 鍵加載策略")
                                policy.load_state_dict(checkpoint[key])
                                print(f"使用 '{key}' 成功加載策略")
                                break
                            except:
                                continue
        # 情況5: 檢查點直接是模型參數
        else:
            print("檢查點不是字典格式，嘗試直接加載")
            policy.load_state_dict(checkpoint)
        
        print("策略加載成功！")
    except Exception as e:
        print(f"所有加載嘗試均失敗: {e}")
        print("正在建立一個新的策略模型...")
    
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
            # 安全地嘗試渲染，如果失敗則輸出警告而不終止程式
            if args.render or args.save_video:
                try:
                    frame = env.render()
                    if args.save_video and frame is not None:
                        frames.append(frame)
                except Exception as e:
                    if episode_length == 0:  # 只在每個回合的第一步輸出警告
                        print(f"警告: 渲染環境時出錯: {e}")
                        print("繼續評估但不進行渲染。如需渲染，請安裝必要的依賴: pip install pygame")
                        # 關閉渲染以避免重複錯誤
                        args.render = False
                        if args.save_video:
                            print("視頻保存功能已停用")
                            args.save_video = False
            
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
    
    # 嘗試安裝所需的依賴
    if args.render or args.save_video:
        try:
            import importlib
            if not importlib.util.find_spec("pygame"):
                print("正在嘗試安裝 pygame...")
                import subprocess
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
                print("pygame 安裝成功！")
        except Exception as e:
            print(f"無法自動安裝 pygame: {e}")
            print("繼續執行但可能無法渲染環境")
    
    # 創建環境
    try:
        env = gym.make(args.env, render_mode="rgb_array" if args.render or args.save_video else None)
    except Exception as e:
        print(f"使用 render_mode 創建環境時出錯: {e}")
        print("嘗試不指定 render_mode 創建環境...")
        env = gym.make(args.env)
        if args.render or args.save_video:
            print("警告: 環境創建時未指定渲染模式，可能無法保存視頻")
    
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