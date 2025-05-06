# GAIL PyTorch實作

基於PyTorch的生成式對抗模仿學習(Generative Adversarial Imitation Learning, GAIL)實作，參考了[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)和[imitation](https://github.com/HumanCompatibleAI/imitation)庫。

## 專案概述

本專案提供了一個系統性的GAIL實作，支援GPU加速和基於PyTorch的深度學習框架。GAIL是一種結合了生成對抗網絡(GAN)和模仿學習的算法，能夠從專家示範中學習策略，而無需獎勵函數。

### 關鍵特性

- **PyTorch實作**：使用PyTorch框架實現，支援GPU加速
- **模組化設計**：清晰的代碼組織，便於擴展和維護
- **支援離散和連續動作空間**：適用於各種強化學習環境
- **可配置的訓練參數**：通過配置文件輕鬆調整訓練設置
- **與Gym環境相容**：可與任何標準Gym環境一起使用
- **專家數據收集工具**：輕鬆從預訓練模型收集專家軌跡

## 代碼風格指南

- 使用PEP 8代碼規範
- 函數和方法應包含清晰的文檔字符串
- 使用類型提示標註函數參數和返回值
- 使用相對導入
- 將複雜的邏輯分解為小型、易於測試的函數

## 編碼模式和約定

- PyTorch作為主要深度學習框架
- 支持GPU加速，但保持CPU兼容性
- 使用`os.path.join`處理文件路徑以保持跨平台兼容性
- 配置應從配置文件加載，而不是硬編碼
- 使用TensorBoard進行實驗跟踪和可視化

## 安裝

### 依賴項

首先，確保您的系統已安裝Python 3.8+和pip。然後安裝所需的依賴項：

```bash
pip install -r requirements.txt
```

### GPU支援

本專案支援GPU加速。要使用GPU，請確保您的系統已安裝CUDA和cuDNN，並且PyTorch已正確配置為使用GPU。

## 使用方法

### 1. 收集專家軌跡

首先，您需要收集專家示範數據。您可以使用提供的腳本從預訓練的模型中收集數據：

```bash
python -m gail_pytorch.scripts.collect_expert_data \
    --env CartPole-v1 \
    --expert_path path/to/expert_model.zip \
    --expert_algo ppo \
    --n_episodes 20 \
    --output_path ./data/expert_trajectories/cartpole_expert.pkl
```

專家模型可以是任何使用stable-baselines3訓練的PPO、SAC或TD3模型。

### 2. 訓練GAIL模型

收集好專家數據後，您可以使用GAIL算法訓練一個模仿策略：

```bash
python -m gail_pytorch.scripts.train_gail \
    --env CartPole-v1 \
    --expert_data ./data/expert_trajectories/cartpole_expert.pkl \
    --total_timesteps 500000 \
    --policy_update_freq 1024 \
    --disc_update_freq 512 \
    --hidden_dim 64 \
    --n_policy_epochs 10 \
    --gae_lambda 0.95 \
    --clip_ratio 0.2 \
    --device cuda
```

主要訓練參數說明：
- `env`: 環境名稱，與收集專家數據時使用的環境相同
- `expert_data`: 專家軌跡數據的路徑
- `total_timesteps`: 訓練的總時間步數
- `policy_update_freq`: 政策更新頻率（每隔多少步更新一次）
- `disc_update_freq`: 判別器更新頻率
- `hidden_dim`: 網絡隱藏層維度
- `n_policy_epochs`: 每次更新策略的訓練輪數
- `gae_lambda`: GAE優勢估計參數
- `clip_ratio`: PPO裁剪比例
- `device`: 使用的設備（'cuda'或'cpu'）

### 3. 使用TensorBoard監控訓練

訓練過程中的指標會自動記錄到TensorBoard中，您可以通過以下命令查看：

```bash
tensorboard --logdir=./data/logs
```

然後在瀏覽器中訪問 http://localhost:6006 查看訓練指標，包括：
- 評估回報和回合長度
- 判別器損失和分類準確率
- 政策損失、價值損失和熵
- 網絡參數分佈等

### 4. 評估訓練的策略

在訓練過程中，GAIL會定期評估策略的性能。您也可以使用訓練腳本中的`evaluate_policy`函數手動評估模型。最終模型會保存在指定的日誌目錄下。

## 專案結構

```
gail_pytorch/
├── agents/           # 強化學習代理定義
├── common/           # 共用函數和類
├── configs/          # 配置文件
├── data/             # 數據目錄
│   ├── expert_trajectories/ # 專家示範數據
│   ├── logs/         # 訓練日誌
│   └── models/       # 保存的模型
├── envs/             # 環境包裝器和自定義環境
├── models/           # 神經網絡模型定義
│   ├── gail.py       # GAIL核心實現
│   └── policy.py     # 策略網絡實現
├── scripts/          # 訓練和評估腳本
│   ├── collect_expert_data.py # 收集專家數據
│   └── train_gail.py # GAIL訓練腳本
└── utils/            # 工具函數
    └── expert_trajectories.py # 專家軌跡處理
```

## 實現細節

GAIL基於Ho & Ermon (2016)的論文《Generative Adversarial Imitation Learning》實現。該算法使用一個判別器來區分專家行為和學習策略的行為，同時訓練策略來欺騙這個判別器。

關鍵組件包括：

- **Discriminator**: 一個神經網絡，輸入狀態-動作對，輸出該對是來自專家還是當前策略的概率
- **Policy Network**: 決定在給定狀態下採取什麼動作的神經網絡
- **GAIL Trainer**: 協調策略和判別器的訓練過程

### 改進的政策訓練

我們的實作採用了PPO風格的政策優化，包括：
- 使用GAE（Generalized Advantage Estimation）計算優勢
- 通過PPO裁剪目標函數穩定訓練
- 多輪策略更新，提高樣本效率
- 價值函數共享，優化獎勵信號

## 未來擴展

- 添加新的模仿學習算法(如AIRL, VAIL)
- 擴展策略網路架構
- 增強數據收集工具
- 添加更多環境適配器
- 實現在線專家示範收集
- 支援多進程數據收集
- 添加更豐富的可視化工具

## 實測效果

在CartPole-v1環境測試中，優化後的GAIL實作能夠從專家示範中成功學習，達到與專家相當的表現（500分滿分）。訓練過程穩定，通常在10分鐘內完成。

## 引用

如果您在研究中使用了本實現，請考慮引用以下論文：

```
@inproceedings{ho2016generative,
  title={Generative adversarial imitation learning},
  author={Ho, Jonathan and Ermon, Stefano},
  booktitle={Advances in Neural Information Processing Systems},
  pages={4565--4573},
  year={2016}
}
```

## 貢獻

歡迎對本專案進行貢獻！請隨時提交問題報告或拉取請求。

## 授權

本專案採用MIT授權。