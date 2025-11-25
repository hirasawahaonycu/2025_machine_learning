# Toy model/Solvable Model Problem

### 主題：基於情感與結構感知的音樂評分模型

## 1. 專案願景 (Vision)
本專案的目標是構建一個「可微分的音樂鑑賞模型」。它不進行生成，而是專注於評價。它將作為未來生成式 AI 的「老師」（Reward Function），引導生成模型創作出更符合人類情感邏輯的作品。

我們不直接定義「好聽」，而是訓練模型識別「結構的合理性」與「情感的準確性」。

## 2. 核心任務定義 (Task Definition)
我們將「評分」這個模糊的任務拆解為兩個可訓練的子任務：

1.  **結構合理性評分 (Structural Integrity Score)**：
    * 任務：區分 *真實的人類樂曲* 與 *打亂/隨機生成的樂曲*。
    * 意義：學習樂理、和聲走向和節奏的規律。
2.  **情感效價預測 (Emotional Valence/Arousal Prediction)**：
    * 任務：給定一段 MIDI/音訊，預測其情感座標（Valence: 積極/消極, Arousal: 激動/平靜）。
    * 意義：讓 AI 理解音樂背後的情緒。

## 3. 數據集策略 (Data Strategy)
這是最關鍵的一步。你原本提到的「準備一組（樂譜，評分）」在現實中很難獲取（誰來給幾萬首曲子打分？）。我們採用**自監督學習 (Self-Supervised Learning)** 和 **弱監督** 替代。

* **數據源**：Lakh MIDI Dataset (LMD) + IMSLP (古典樂 MIDI)。
* **數據預處理**：將 MIDI 轉換為 **Piano Roll（鋼琴捲簾圖）**。這是一個二維矩陣 $X \in \{0,1\}^{T \times 88}$ （時間 $\times$ 音高），類似於圖片。

## 4. 模型架構設計 (Model Architecture)
我們放棄單純的回歸模型，改用混合架構，結合視覺（和聲結構）與時序（旋律走向）。

**模型名稱：MuseNet-Evaluator**

* **輸入層**：Piano Roll 片段 (例如 16 小節)。
* **特徵提取層 (Backbone)**：
    * **CNN (卷積層)**：負責提取局部的和弦特徵（例如：C 大調和弦的形狀在圖像上是有規律的）。
    * **Bi-LSTM / Transformer Encoder**：負責提取時間上的旋律流向（例如：解決音的進行，張力的釋放）。
* **輸出層 (Heads)**：
    1.  **Head A (Binary Classifier)**: 輸出 $P(\text{Real} | x)$ ，判斷這是人類作品還是噪音。
    2.  **Head B (Regression)**: 輸出 $(v, a)$ 情感向量。

## 5. 訓練方法 (Training Methodology)
這是對你原本「擴散模型思想」和「回歸方案」的優化整合。

### 階段一：自監督預訓練 (讓 AI 懂樂理)
我們不需要人工標註，通過**對比學習 (Contrastive Learning)** 訓練：
* **正樣本**：取自數據集的真實人類片段。
* **負樣本**：
    * *Random Noise*：完全隨機的音符（學習區分噪音）。
    * *Pitch Shifted Dissonance*：將原始好聽的和弦中的某幾個音符隨機移位，製造不協和音（學習分辨走音）。
    * *Time Scramble*：將原來的小節順序打亂（學習曲式結構）。
* **損失函數**：使用 InfoNCE Loss 或 Binary Cross Entropy，迫使模型給「正樣本」打高分，給「負樣本」打低分。

### 階段二：情感微調 (讓 AI 懂情緒)
* 利用帶有情感標籤的小型數據集（如 EMOPIA Dataset，包含簡單的 4 類情感標註：快樂、憤怒、悲傷、平靜）。
* 凍結 CNN 特徵層，只訓練情感回歸頭（Regression Head）。

## 6. 預期成果與驗證 (Evaluation)
如何證明這個簡化模型成功了？

1.  **Turing Test for Music (圖靈測試)**：
    * 給模型聽巴哈的《平均律》和一段隨機生成的 AI 音樂，模型的評分必須顯著傾向於巴哈。
2.  **情感分類準確率**：
    * 給模型一首悲傷的音樂，它不能預測出「快樂」。
3.  **可視化 (Saliency Map)**：
    * 類似於圖像識別的熱力圖，我們應該能看到模型關注 Piano Roll 上的哪些部分。如果它關注到了「不協和音程」並因此扣分，說明它真的學會了樂理。

## 7. 下一步：連接到生成模型 (The Future)
一旦這個 MuseCritic 訓練完成，它就可以作為 **Verifier** 嵌入到你未來的生成模型中：
* **拒絕採樣 (Rejection Sampling)**：生成器生成 100 段旋律，MuseCritic 挑出分數最高的 1 段輸出。
* **強化學習 (RL)**：將 MuseCritic 的評分作為 Reward，通過 PPO 演算法優化生成器。
