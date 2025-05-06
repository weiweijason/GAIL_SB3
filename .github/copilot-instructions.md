<!-- 
  使用此文件為Copilot提供工作區特定的自定義指令。
  有關詳細信息，請訪問https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file 
-->

# GAIL PyTorch專案的Copilot指導

這是一個基於PyTorch的生成式對抗模仿學習(GAIL)實作專案。本專案使用模塊化設計，支持GPU加速，並參考了stable-baselines3和imitation庫。

## 專案架構

專案採用以下模塊化結構：
- `models/`: 包含GAIL算法和策略網路的核心實現
- `agents/`: 強化學習代理定義
- `utils/`: 工具函數，包括專家軌跡處理
- `configs/`: 配置文件
- `scripts/`: 訓練和數據收集腳本
- `envs/`: 環境相關代碼

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

## 擴展專案

擴展此專案時，請考慮以下方向：
- 添加新的模仿學習算法(如AIRL, VAIL)
- 擴展策略網路架構
- 增強數據收集工具
- 添加更多環境適配器
- 實現在線專家示範收集