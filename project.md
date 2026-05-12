# **Reward Design in Reinforcement Learning: A Case Study on 2048**

---

# **1. Introduction**

## **A. Attention Getter**

2048 是一款規則簡單但策略深度很高的益智遊戲。
玩家每一步只能選擇四個方向移動，但要取得高分，往往需要長期規劃與盤面管理。

因此，2048 同時具有：

* **簡單的行動空間**
* **複雜的策略決策**

這使它成為研究 **強化學習（Reinforcement Learning, RL）** 的一個有趣平台。

---

## **B. Motivation**

近年來，強化學習在許多領域展現出優秀表現，例如：

* 遊戲（AlphaGo、Atari）
* 機器人控制
* 自動決策系統

然而，RL 的學習效果往往 **高度依賴 reward 設計**。

在許多實際問題中，reward 並不是每一步都能得到明確回饋，而是 **延遲出現（delayed reward）**。

2048 正是一個典型例子：

* 分數只在 **tile 合併時** 才會增加
* 良好的策略（例如保持盤面整齊）通常 **沒有直接 reward**

因此，僅依賴原始分數作為 reward，可能無法有效引導 agent 學習良好的策略。

---

## **C. Challenge**

在 2048 環境中，強化學習面臨幾個挑戰：

1️⃣ **Reward 稀疏（Sparse Reward）**

* 分數只在合併 tile 時出現

2️⃣ **策略價值難以反映**

* 保持空位、維持單調盤面等策略沒有直接回饋

3️⃣ **容易出現短視策略**

* Agent 可能只追求眼前合併，而忽略長期盤面管理

這些因素都可能導致：

* 學習速度變慢
* 策略品質不穩定

---

## **D. Research Objective**

本研究的核心問題為：

> **Reward 設計是否會影響強化學習在 2048 中的學習效果？**

特別關注：

* Reward Shaping 是否能

  * 加速學習
  * 提升最終表現
  * 引導更穩定的策略

---

## **E. Method Overview**

本研究使用 **Deep Q-Network (DQN)** 作為強化學習方法。

為了確保實驗公平：

* **模型架構完全固定**
* **訓練設定完全相同**
* 唯一改變的因素是 **reward 設計**

所有模型皆使用 **PyTorch** 實作。

---

## **F. Experiments**

我們設計四種不同 agent 進行比較：

| Agent    | 說明                  |
| -------- | ------------------- |
| Random   | 隨機行動，不進行學習          |
| Baseline | 使用原始分數作為 reward     |
| + Empty  | 加入空位數 reward        |
| + Full   | 使用完整 reward shaping |

透過比較不同 reward 設計對學習結果的影響，分析 reward shaping 的效果。

---

## **G. Expected Findings**

我們預期：

* Reward shaping 能 **加速學習過程**
* 能 **提高最終分數**
* Agent 能學習到 **更穩定的盤面策略**

本研究希望說明：

> **Reward 設計在強化學習中具有關鍵影響。**

---

# **2. Related Work**

2048 的求解方法大致可以分為兩類：

---

## **1. Search-based Methods**

許多傳統方法使用 **Expectimax search**。

其特點為：

* 透過搜尋預測未來盤面
* 不需要學習

然而這類方法：

* 計算成本較高
* 無法從經驗中學習策略

---

## **2. Reinforcement Learning Approaches**

近年研究開始使用 RL 來解決 2048。

其中 **Deep Q-Network (DQN)** 能夠：

* 處理高維狀態空間
* 從遊戲經驗中學習策略

然而過去研究指出：

* 使用 **原始 score 作為 reward** 往往效果有限
* 適當的 **reward shaping** 可以改善學習效果

例如加入：

* 空位數
* monotonicity
* tile 位置結構

這些設計可以幫助 agent 學習更好的盤面策略。

---

# **3. Proposed Design**

---

# **3.1 Environment**

本研究使用標準 2048 遊戲環境。

環境設定如下：

| 項目          | 說明         |
| ----------- | ---------- |
| Grid size   | 4 × 4      |
| Actions     | 上、下、左、右    |
| Tile spawn  | 每步生成 2 或 4 |
| Termination | 無法再移動      |

遊戲的目標為：

> **盡可能獲得高分並產生更大的 tile**

---

# **3.2 Model**

我們使用 **Deep Q-Network (DQN)** 作為學習模型。

模型架構採用簡單的 **Multi-Layer Perceptron (MLP)**。

主要設定：

| 元件            | 設定                |
| ------------- | ----------------- |
| Network       | MLP               |
| Optimizer     | Adam              |
| Loss          | Huber loss        |
| Replay Buffer | Experience Replay |
| Exploration   | ε-greedy          |

重要的是：

> **所有實驗使用完全相同的模型設定**

因此性能差異主要來自 **reward 設計**。

---

# **3.3 Reward Design（核心）**

本研究設計三種不同 reward 方式。

---

## **(1) Baseline Reward**

最基本的設計：

Reward 只使用遊戲分數。

$R = r_{score}$

其中 $r_{score}$ 為 tile 合併所得到的分數。

此設計對應於 **Sparse Reward** 的情況。

---

## **(2) Partial Reward Shaping**

在 baseline 的基礎上加入 **空位數 reward**：

$R = r_{score} + \alpha r_{empty}$

其中：

* $r_{empty}$：盤面空格數量
* $\alpha$：權重參數

此設計鼓勵 agent：

* 保持盤面有更多可操作空間
* 避免盤面過早填滿

---

## **(3) Full Reward Shaping**

完整 reward 設計：

$R = r_{score} + \alpha r_{empty} + \beta r_{corner} + \gamma r_{monotonic}$

其中：

* **Corner reward**：鼓勵最大 tile 保持在角落。
* **Monotonic reward**：勵盤面呈現單調排列，有助於形成穩定策略。

透過這些額外 reward，agent 能更容易學習到良好的盤面管理策略。

---

# **3.4 Evaluation Metrics**

我們使用以下指標評估模型表現：

| 指標             | 說明          |
| -------------- | ----------- |
| Average Score  | 平均遊戲分數      |
| Max Tile       | 能達到的最大 tile |
| Training Curve | 訓練過程分數變化    |

此外也會比較：

* **Random Agent**
* **不同 reward 設計**

以分析 reward shaping 對學習效果的影響。
