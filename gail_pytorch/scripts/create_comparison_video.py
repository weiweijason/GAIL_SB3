# 建議檔案路徑: gail_pytorch/scripts/create_comparison_video.py
"""
創建 GAIL 模型與專家行為的並排比較視頻。

此腳本加載 GAIL 模型和專家策略，並創建一個並排視頻來直觀比較兩者的行為。
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
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description="創建 GAIL 模型與專家行為的並排比較視頻")
    
    parser.add_argument("--gail_model_path", type=str, required=True,
                      help="已訓練的 GAIL 模型路徑")
    parser.add_argument("--expert_policy_path", type=str,
                      help="專家政策模型路徑（如果有）")
    parser.add_argument("--expert_data", type=str, required=True,
                      help="專家軌跡數據路徑（用於評估或作為專家政策的替代）")
    parser.add_argument("--env", type=str, default="CartPole-v1",
                      help="Gym 環境名稱")
    parser.add_argument("--n_episodes", type=int, default=3,
                      help="比較的回合數")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                      help="最大回合長度")
    parser.add_argument("--seed", type=int, default=0,
                      help="隨機種子")
    parser.add_argument("--output_path", type=str, default="./data/videos/comparison",
                      help="輸出視頻的保存路徑")
    parser.add_argument("--fps", type=int, default=30,
                      help="視頻的幀率")
    parser.add_argument("--hidden_dims", type=str, default="64,64",
                      help="模型的隱藏層維度，用逗號分隔，例如：64,64")
    parser.add_argument("--width", type=int, default=1280,
                      help="視頻寬度")
    parser.add_argument("--height", type=int, default=480,
                      help="視頻高度")
    parser.add_argument("--device", type=str, default="cuda",
                      help="運行設備 (cuda 或 cpu)")
    
    return parser.parse_args()


def load_gail_policy(model_path, env, device, hidden_dims_str="64,64"):
    """加載 GAIL 政策模型。"""
    # 解析隱藏層維度
    hidden_dims = tuple(int(dim) for dim in hidden_dims_str.split(','))
    print(f"使用隱藏層維度: {hidden_dims}")
    
    # 確定動作空間類型
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        action_dim = env.action_space.n
    else:
        is_discrete = False
        action_dim = env.action_space.shape[0]
    
    # 獲取狀態維度
    state_dim = env.observation_space.shape[0]
    
    # 創建政策網絡
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
    
    # 加載模型參數
    print(f"從 {model_path} 加載模型...")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            if "policy" in checkpoint:
                policy.load_state_dict(checkpoint["policy"])
                print("成功加載 GAIL 政策模型")
            elif isinstance(checkpoint, dict) and len(checkpoint) > 0:
                # 嘗試找到政策參數
                for key in checkpoint.keys():
                    if "policy" in key.lower():
                        policy.load_state_dict(checkpoint[key])
                        print(f"使用 '{key}' 成功加載 GAIL 政策模型")
                        break
                else:
                    # 直接嘗試加載
                    try:
                        policy.load_state_dict(checkpoint)
                        print("直接加載 GAIL 政策模型成功")
                    except Exception as e:
                        print(f"加載 GAIL 政策失敗: {e}")
                        raise
        else:
            # 直接是政策的狀態字典
            policy.load_state_dict(checkpoint)
            print("直接加載 GAIL 政策模型成功")
            
    except Exception as e:
        print(f"加載模型時出錯: {e}")
        raise
    
    policy.eval()  # 設置為評估模式
    return policy, is_discrete


def run_episodes(env, policy, is_discrete, n_episodes, max_steps, seed=None, model_name="模型"):
    """運行多個回合並收集渲染幀。"""
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
            # 渲染當前狀態
            frame = env.render()
            frames.append(frame)
            
            # 從政策獲取動作
            if is_discrete:
                action, _, _ = policy.get_action(state, deterministic=True)
            else:
                action, _, _ = policy.get_action(state, deterministic=True)
            
            # 在環境中執行動作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新狀態和回報
            state = next_state
            ep_return += reward
            step += 1
        
        all_frames.append(frames)
        all_returns.append(ep_return)
        all_lengths.append(step)
        
        print(f"{model_name} 回合 {i+1}/{n_episodes} - 回報: {ep_return:.2f}, 長度: {step}")
    
    avg_return = np.mean(all_returns)
    avg_length = np.mean(all_lengths)
    print(f"{model_name} 平均回報: {avg_return:.2f}, 平均長度: {avg_length:.2f}")
    
    return all_frames, all_returns, all_lengths


def replay_expert_trajectories(env, expert_data, n_episodes, max_steps, seed=None):
    """從專家軌跡數據中重放專家行為。"""
    all_frames = []
    all_returns = []
    all_lengths = []
    
    # 從專家數據中提取軌跡
    # 假設專家數據包含連續的時間步，並且可以重建完整的回合
    states = expert_data["states"]
    actions = expert_data["actions"]
    rewards = expert_data.get("rewards", [])
    dones = expert_data.get("dones", [])
    
    # 取得回合分界點
    episode_ends = [i for i, done in enumerate(dones) if done] if len(dones) > 0 else []
    if not episode_ends:
        # 如果沒有明確的回合結束，嘗試從其他信息推斷
        if "episode_lengths" in expert_data:
            lengths = expert_data["episode_lengths"]
            episode_ends = []
            current_end = 0
            for length in lengths:
                current_end += length
                episode_ends.append(current_end - 1)  # 轉為0索引
    
    # 如果仍然無法確定回合邊界，假設所有數據都是一個回合
    if not episode_ends:
        episode_ends = [len(states) - 1]
    
    # 選擇要重放的回合
    if seed is not None:
        np.random.seed(seed)
    
    if len(episode_ends) <= n_episodes:
        selected_episodes = range(len(episode_ends))
    else:
        selected_episodes = np.random.choice(len(episode_ends), n_episodes, replace=False)
    
    # 對每個選定的回合進行重放
    for ep_idx in selected_episodes:
        frames = []
        start_idx = 0 if ep_idx == 0 else episode_ends[ep_idx - 1] + 1
        end_idx = episode_ends[ep_idx]
        
        # 設置環境到初始狀態
        state, _ = env.reset()
        
        # 重放這個回合
        ep_return = 0
        episode_length = min(end_idx - start_idx + 1, max_steps)
        
        for i in range(episode_length):
            idx = start_idx + i
            if idx > end_idx:
                break
                
            # 渲染當前狀態
            frame = env.render()
            frames.append(frame)
            
            # 獲取專家動作並執行
            action = actions[idx]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 更新狀態和回報
            state = next_state
            ep_return += reward
            
            if done:
                break
        
        all_frames.append(frames)
        all_returns.append(ep_return)
        all_lengths.append(episode_length)
        
        print(f"專家回合 {ep_idx+1} - 回報: {ep_return:.2f}, 長度: {episode_length}")
    
    avg_return = np.mean(all_returns)
    avg_length = np.mean(all_lengths)
    print(f"專家平均回報: {avg_return:.2f}, 平均長度: {avg_length:.2f}")
    
    return all_frames, all_returns, all_lengths


def load_expert_policy(policy_path, env, device, hidden_dims_str="64,64"):
    """加載專家政策模型（如果可用）。"""
    # 與 load_gail_policy 函數類似
    return load_gail_policy(policy_path, env, device, hidden_dims_str)


def create_side_by_side_frame(gail_frame, expert_frame, gail_info, expert_info, width=1280, height=480):
    """創建並排比較幀。"""
    # 創建一個新的圖形
    fig = plt.figure(figsize=(width/100, height/100), dpi=100)
    canvas = FigureCanvas(fig)
    
    # 確保兩幀具有相同的大小
    if gail_frame.shape != expert_frame.shape:
        # 調整大小使兩幀尺寸一致
        from skimage.transform import resize
        min_height = min(gail_frame.shape[0], expert_frame.shape[0])
        min_width = min(gail_frame.shape[1], expert_frame.shape[1])
        gail_frame = resize(gail_frame, (min_height, min_width), preserve_range=True).astype(np.uint8)
        expert_frame = resize(expert_frame, (min_height, min_width), preserve_range=True).astype(np.uint8)
    
    # 創建並排佈局
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(gail_frame)
    ax1.set_title(f"GAIL 模型\n{gail_info}")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(expert_frame)
    ax2.set_title(f"專家行為\n{expert_info}")
    ax2.axis('off')
    
    fig.tight_layout()
    
    # 轉換為圖像
    canvas.draw()
    comparison_frame = np.array(canvas.renderer.buffer_rgba())
    
    plt.close(fig)
    return comparison_frame


def create_comparison_video(gail_frames_list, expert_frames_list, gail_returns, expert_returns, 
                           gail_lengths, expert_lengths, output_path, fps=30, width=1280, height=480):
    """創建 GAIL 模型與專家行為的並排比較視頻。"""
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_path, f"comparison_{timestamp}.mp4")
    
    # 確保我們有相同數量的回合
    n_episodes = min(len(gail_frames_list), len(expert_frames_list))
    
    # 為每個回合創建一個視頻
    for ep in range(n_episodes):
        gail_frames = gail_frames_list[ep]
        expert_frames = expert_frames_list[ep]
        
        # 確保我們有足夠的幀進行比較
        max_frames = max(len(gail_frames), len(expert_frames))
        
        # 準備比較幀
        comparison_frames = []
        
        # 創建信息字符串
        gail_info = f"回報: {gail_returns[ep]:.2f}, 長度: {gail_lengths[ep]}"
        expert_info = f"回報: {expert_returns[ep]:.2f}, 長度: {expert_lengths[ep]}"
        
        for i in range(max_frames):
            # 獲取 GAIL 幀 (如果可用，否則使用最後一幀)
            gail_idx = min(i, len(gail_frames) - 1)
            gail_frame = gail_frames[gail_idx]
            
            # 獲取專家幀 (如果可用，否則使用最後一幀)
            expert_idx = min(i, len(expert_frames) - 1)
            expert_frame = expert_frames[expert_idx]
            
            # 創建並排比較
            comparison = create_side_by_side_frame(
                gail_frame, expert_frame, gail_info, expert_info, width, height
            )
            comparison_frames.append(comparison)
        
        # 保存這個回合的視頻
        ep_output_file = os.path.join(output_path, f"comparison_ep{ep+1}_{timestamp}.mp4")
        imageio.mimsave(ep_output_file, comparison_frames, fps=fps)
        print(f"回合 {ep+1} 比較視頻已保存到 {ep_output_file}")
    
    # 創建所有回合的合併視頻
    all_comparison_frames = []
    for ep in range(n_episodes):
        gail_frames = gail_frames_list[ep]
        expert_frames = expert_frames_list[ep]
        
        # 確保我們有足夠的幀進行比較
        max_frames = max(len(gail_frames), len(expert_frames))
        
        # 創建信息字符串
        gail_info = f"回合 {ep+1}: 回報={gail_returns[ep]:.2f}, 長度={gail_lengths[ep]}"
        expert_info = f"回合 {ep+1}: 回報={expert_returns[ep]:.2f}, 長度={expert_lengths[ep]}"
        
        for i in range(max_frames):
            # 獲取 GAIL 幀 (如果可用，否則使用最後一幀)
            gail_idx = min(i, len(gail_frames) - 1)
            gail_frame = gail_frames[gail_idx]
            
            # 獲取專家幀 (如果可用，否則使用最後一幀)
            expert_idx = min(i, len(expert_frames) - 1)
            expert_frame = expert_frames[expert_idx]
            
            # 創建並排比較
            comparison = create_side_by_side_frame(
                gail_frame, expert_frame, gail_info, expert_info, width, height
            )
            all_comparison_frames.append(comparison)
        
        # 添加一個短暫的黑幀作為回合之間的分隔符
        if ep < n_episodes - 1:
            separator = np.zeros((height, width, 4), dtype=np.uint8)
            for _ in range(int(fps/2)):  # 0.5秒的分隔
                all_comparison_frames.append(separator)
    
    # 保存合併視頻
    imageio.mimsave(output_file, all_comparison_frames, fps=fps)
    print(f"所有回合的比較視頻已保存到 {output_file}")
    
    return output_file


def main(args):
    """主函數。"""
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 創建環境（帶渲染選項）
    try:
        env = gym.make(args.env, render_mode="rgb_array")
    except Exception as e:
        print(f"創建帶渲染的環境時出錯: {e}")
        print("嘗試不指定 render_mode 創建環境...")
        env = gym.make(args.env)
    
    # 加載 GAIL 政策
    gail_policy, is_discrete = load_gail_policy(
        args.gail_model_path, env, args.device, args.hidden_dims
    )
    
    # 運行 GAIL 模型並收集幀
    print("\n=== 運行 GAIL 模型 ===")
    gail_frames, gail_returns, gail_lengths = run_episodes(
        env, gail_policy, is_discrete, args.n_episodes, args.max_ep_len, args.seed, "GAIL"
    )
    
    # 加載專家數據
    expert_data = load_expert_trajectories(args.expert_data)
    print(f"\n已加載專家數據，包含 {len(expert_data['states'])} 個時間步")
    
    # 如果提供了專家政策路徑，則加載專家政策
    if args.expert_policy_path and os.path.exists(args.expert_policy_path):
        print("\n=== 加載專家政策 ===")
        expert_policy, expert_is_discrete = load_expert_policy(
            args.expert_policy_path, env, args.device, args.hidden_dims
        )
        
        print("\n=== 運行專家政策 ===")
        expert_frames, expert_returns, expert_lengths = run_episodes(
            env, expert_policy, expert_is_discrete, args.n_episodes, 
            args.max_ep_len, args.seed + 100, "專家"
        )
    else:
        # 使用專家軌跡數據重放專家行為
        print("\n=== 重放專家軌跡 ===")
        expert_frames, expert_returns, expert_lengths = replay_expert_trajectories(
            env, expert_data, args.n_episodes, args.max_ep_len, args.seed + 100
        )
    
    # 創建比較視頻
    print("\n=== 創建比較視頻 ===")
    output_file = create_comparison_video(
        gail_frames, expert_frames, gail_returns, expert_returns,
        gail_lengths, expert_lengths, args.output_path, args.fps,
        args.width, args.height
    )
    
    print(f"\n比較視頻已成功創建: {output_file}")
    print(f"GAIL 平均回報: {np.mean(gail_returns):.2f} ± {np.std(gail_returns):.2f}")
    print(f"專家平均回報: {np.mean(expert_returns):.2f} ± {np.std(expert_returns):.2f}")
    
    # 關閉環境
    env.close()


if __name__ == "__main__":
    args = parse_args()
    main(args)