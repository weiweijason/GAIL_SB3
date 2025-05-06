# 建議文件路徑: gail_pytorch/scripts/debug_model.py
"""
調試GAIL模型加載和評估的腳本。

此腳本用於詳細診斷模型加載和評估過程中的問題。
"""
import os
import sys
import argparse
import numpy as np
import torch
import gym

from gail_pytorch.models.policy import DiscretePolicy, ContinuousPolicy

def parse_args():
    """解析命令行參數。"""
    parser = argparse.ArgumentParser(description="調試GAIL模型")
    
    parser.add_argument("--model_path", type=str, required=True,
                        help="已訓練模型的路徑")
    parser.add_argument("--env", type=str, default="CartPole-v1", 
                        help="Gym環境名稱")
    parser.add_argument("--max_ep_len", type=int, default=1000,
                        help="最大回合長度")
    parser.add_argument("--seed", type=int, default=0,
                        help="隨機種子")
    parser.add_argument("--device", type=str, default="cuda",
                        help="運行設備 (cuda 或 cpu)")
    
    return parser.parse_args()

def debug_model_structure(model_path, device):
    """檢查模型檔案結構並打印詳細信息。"""
    print(f"\n=== 檢查模型檔案結構 ===")
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        print(f"模型類型: {type(checkpoint)}")
        
        if isinstance(checkpoint, dict):
            print(f"模型包含的鍵: {list(checkpoint.keys())}")
            
            for key, value in checkpoint.items():
                if isinstance(value, dict):
                    print(f"  - {key}: 字典，包含 {len(value)} 個項目")
                elif isinstance(value, torch.Tensor):
                    print(f"  - {key}: 張量，形狀 {value.shape}")
                else:
                    print(f"  - {key}: {type(value)}")
        else:
            print("模型不是字典格式，無法查看內部結構")
            
    except Exception as e:
        print(f"檢查模型結構時出錯: {e}")

def run_single_episode(policy, env, max_steps=1000, seed=None, debug=True):
    """運行單個回合並打印詳細調試信息。"""
    returns = 0
    length = 0
    
    # 重置環境
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()
    
    done = False
    
    if debug:
        print("\n=== 回合詳細信息 ===")
    
    while not done and length < max_steps:
        # 從策略獲取動作
        if isinstance(policy, DiscretePolicy):
            action_logits, _ = policy(torch.FloatTensor(state).unsqueeze(0).to(policy.device))
            action_probs = torch.nn.functional.softmax(action_logits, dim=-1)
            
            if debug and length == 0:
                print(f"動作概率: {action_probs.detach().cpu().numpy()}")
            
            if torch.isnan(action_probs).any() or torch.isinf(action_probs).any():
                print(f"警告: 動作概率包含NaN或Inf值!")
                
            action, log_prob, _ = policy.get_action(state, deterministic=True)
            
            if debug and length < 5:  # 只打印前5步以避免信息過多
                print(f"步數 {length}: 狀態 {state}, 動作 {action}, 動作概率 {action_probs.detach().cpu().numpy()}")
        else:
            action_means, action_log_stds, _ = policy(torch.FloatTensor(state).unsqueeze(0).to(policy.device))
            
            if debug and length == 0:
                print(f"動作均值: {action_means.detach().cpu().numpy()}, 動作對數標準差: {action_log_stds.detach().cpu().numpy()}")
            
            if torch.isnan(action_means).any() or torch.isinf(action_means).any():
                print(f"警告: 動作均值包含NaN或Inf值!")
                
            action, log_prob, _ = policy.get_action(state, deterministic=True)
            
            if debug and length < 5:
                print(f"步數 {length}: 狀態 {state}, 動作 {action}")
        
        # 執行動作
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 更新值
        state = next_state
        returns += reward
        length += 1
    
    if debug:
        print(f"回合結束: 總回報 {returns}，長度 {length}，是否自然結束: {done}")
    
    return returns, length, done

def main(args):
    """主調試函數。"""
    print("\n=== GAIL模型調試工具 ===")
    print(f"模型路徑: {args.model_path}")
    print(f"環境: {args.env}")
    print(f"設備: {args.device}")
    
    # 設置隨機種子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 詳細檢查模型結構
    debug_model_structure(args.model_path, args.device)
    
    # 創建環境
    env = gym.make(args.env)
    
    # 確定動作空間和狀態空間
    if isinstance(env.action_space, gym.spaces.Discrete):
        is_discrete = True
        action_dim = env.action_space.n
        print(f"\n離散動作空間: {action_dim} 個可能的動作")
    else:
        is_discrete = False
        action_dim = env.action_space.shape[0]
        action_low = env.action_space.low
        action_high = env.action_space.high
        print(f"\n連續動作空間: 維度 {action_dim}, 範圍 {action_low} 到 {action_high}")
    
    state_dim = env.observation_space.shape[0]
    print(f"狀態空間: 維度 {state_dim}")
    
    # 創建策略網絡
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
    
    # 加載模型
    try:
        print("\n嘗試加載模型...")
        checkpoint = torch.load(args.model_path, map_location=args.device)
        
        loaded = False
        
        if isinstance(checkpoint, dict):
            # 常規加載嘗試
            potential_keys = ['policy', 'model', 'network', 'actor', 'state_dict']
            for key in potential_keys:
                if key in checkpoint:
                    try:
                        policy.load_state_dict(checkpoint[key])
                        print(f"成功使用 '{key}' 鍵加載模型")
                        loaded = True
                        break
                    except Exception as e:
                        print(f"使用 '{key}' 加載失敗: {e}")
            
            # 針對GAIL特殊處理 - 嘗試查找分離的策略文件
            if not loaded and 'discriminator' in checkpoint:
                print("\n檢測到GAIL模型格式，但缺少策略網絡參數")
                print("嘗試查找其他可能的策略檔案...")
                
                # 嘗試從相同目錄加載不同命名的策略檔案
                model_dir = os.path.dirname(args.model_path)
                base_name = os.path.basename(args.model_path).replace('gail_model', '').replace('.pt', '')
                
                potential_policy_files = [
                    os.path.join(model_dir, f"policy_model{base_name}.pt"),
                    os.path.join(model_dir, f"policy{base_name}.pt"),
                    os.path.join(model_dir, f"actor{base_name}.pt"),
                    # 檢查更早的檢查點
                    os.path.join(model_dir, "gail_model_*00000.pt"),
                    os.path.join(model_dir, "policy_*.pt")
                ]
                
                # 列出目錄中的所有檔案
                try:
                    all_files = os.listdir(model_dir)
                    print(f"目錄 {model_dir} 中的檔案: {all_files}")
                    
                    # 檢查是否有任何可能的策略檔案
                    policy_candidates = [f for f in all_files if "policy" in f.lower() or "actor" in f.lower() or ("model" in f.lower() and "gail" not in f.lower())]
                    
                    if policy_candidates:
                        print(f"找到可能的策略檔案: {policy_candidates}")
                        for policy_file in policy_candidates:
                            try:
                                full_path = os.path.join(model_dir, policy_file)
                                print(f"嘗試加載 {full_path}...")
                                policy_checkpoint = torch.load(full_path, map_location=args.device)
                                
                                if isinstance(policy_checkpoint, dict):
                                    for key in potential_keys:
                                        if key in policy_checkpoint:
                                            try:
                                                policy.load_state_dict(policy_checkpoint[key])
                                                print(f"成功從 {policy_file} 使用 '{key}' 鍵加載策略")
                                                loaded = True
                                                break
                                            except Exception as e:
                                                print(f"從 {policy_file} 使用 '{key}' 加載失敗: {e}")
                                    
                                    if not loaded:
                                        try:
                                            policy.load_state_dict(policy_checkpoint)
                                            print(f"成功直接從 {policy_file} 加載策略")
                                            loaded = True
                                            break
                                        except Exception as e:
                                            print(f"直接從 {policy_file} 加載失敗: {e}")
                                else:
                                    try:
                                        policy.load_state_dict(policy_checkpoint)
                                        print(f"成功從 {policy_file} 加載策略")
                                        loaded = True
                                        break
                                    except Exception as e:
                                        print(f"從 {policy_file} 加載失敗: {e}")
                            except Exception as e:
                                print(f"處理 {policy_file} 時出錯: {e}")
                
                except Exception as e:
                    print(f"列出目錄內容時出錯: {e}")
                
                # 使用GAIL訓練腳本的格式創建新的策略模型路徑
                if not loaded:
                    # 尋找訓練過程中保存的中間檢查點
                    import glob
                    checkpoint_pattern = os.path.join(model_dir, "gail_model_*.pt")
                    checkpoints = glob.glob(checkpoint_pattern)
                    
                    if checkpoints:
                        # 按照修改時間排序，優先嘗試最新的檢查點
                        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                        print(f"找到 {len(checkpoints)} 個檢查點，嘗試依次加載...")
                        
                        for checkpoint_path in checkpoints[:5]:  # 只嘗試最新的5個檢查點
                            if checkpoint_path != args.model_path:  # 避免重複嘗試當前檔案
                                try:
                                    print(f"嘗試從檢查點 {checkpoint_path} 加載策略...")
                                    alt_checkpoint = torch.load(checkpoint_path, map_location=args.device)
                                    
                                    if isinstance(alt_checkpoint, dict) and 'policy' in alt_checkpoint:
                                        try:
                                            policy.load_state_dict(alt_checkpoint['policy'])
                                            print(f"成功從檢查點 {checkpoint_path} 加載策略")
                                            loaded = True
                                            break
                                        except Exception as e:
                                            print(f"從檢查點 {checkpoint_path} 加載失敗: {e}")
                                except Exception as e:
                                    print(f"處理檢查點 {checkpoint_path} 時出錯: {e}")
            
            # 如果所有嘗試都失敗，創建一個隨機策略並提供警告
            if not loaded:
                if 'discriminator' in checkpoint:
                    print("\n警告: 無法找到有效的策略模型。檢測到GAIL模型，但只包含判別器而無策略參數。")
                    print("這表明模型可能未被正確保存，或者策略參數被保存在單獨的檔案中。")
                    print("建議檢查訓練腳本中模型保存的邏輯，確保完整保存策略參數。")
                    
                    # 作為臨時解決方案，嘗試直接使用判別器的參數
                    try:
                        disc_params = checkpoint['discriminator']
                        print("嘗試直接從判別器參數初始化策略...")
                        
                        # 統計判別器參數的大小並打印
                        disc_size = sum(p.numel() for p in disc_params.values())
                        print(f"判別器參數總數: {disc_size}")
                        
                        # 打印判別器的層結構
                        for name, param in disc_params.items():
                            if isinstance(param, torch.Tensor):
                                print(f"  - {name}: {param.shape}")
                    except Exception as e:
                        print(f"分析判別器參數時出錯: {e}")
                else:
                    print("\n警告: 無法加載任何策略模型。將使用隨機初始化的策略進行評估。")
                print("模型評估結果可能不准確，反映的是隨機策略而非訓練後的策略。")
        else:
            try:
                policy.load_state_dict(checkpoint)
                print("成功直接加載模型")
                loaded = True
            except Exception as e:
                print(f"直接加載失敗: {e}")
                print("\n警告: 無法加載模型，將使用隨機初始化的策略進行評估。")
            
    except Exception as e:
        print(f"加載模型時出錯: {e}")
        return
    
    # 設置為評估模式
    policy.eval()
    
    # 執行單個回合並收集調試信息
    print("\n執行單個回合進行詳細調試...")
    returns, length, done = run_single_episode(policy, env, max_steps=args.max_ep_len, seed=args.seed, debug=True)
    
    print("\n=== 回合摘要 ===")
    print(f"總回報: {returns}")
    print(f"回合長度: {length}")
    print(f"是否自然結束: {done}")
    
    # 執行多個回合以計算平均表現
    n_eval_episodes = 5
    all_returns = []
    all_lengths = []
    
    print(f"\n執行 {n_eval_episodes} 個回合進行評估...")
    
    for i in range(n_eval_episodes):
        returns, length, _ = run_single_episode(policy, env, max_steps=args.max_ep_len, seed=args.seed+i, debug=False)
        all_returns.append(returns)
        all_lengths.append(length)
        print(f"回合 {i+1}/{n_eval_episodes} - 回報: {returns:.2f}, 長度: {length}")
    
    print("\n=== 評估結果 ===")
    print(f"平均回報: {np.mean(all_returns):.2f} ± {np.std(all_returns):.2f}")
    print(f"平均回合長度: {np.mean(all_lengths):.2f} ± {np.std(all_lengths):.2f}")
    print(f"最高回報: {max(all_returns):.2f}")
    print(f"最低回報: {min(all_returns):.2f}")
    
    # 關閉環境
    env.close()

if __name__ == "__main__":
    args = parse_args()
    main(args)