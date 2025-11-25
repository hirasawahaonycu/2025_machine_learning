# Week 1

### 問題 1：Why does adding a nonlinear layer after each linear layer make the model better?

**（為什麼在每個線性層之後添加非線性層會讓模型更好？）**

  * **AI 研究解答：**
    這涉及到神經網路的 **「通用近似定理」（Universal Approximation Theorem）**。
    如果神經網路只有線性層，無論疊加多少層，數學上它仍然等同於**單一個**線性變換（因為矩陣乘法的結合律： $W_2(W_1x) = (W_2W_1)x = W_{new}x$ ）。
    加入非線性激活函數（如 Sigmoid, ReLU）打破了這種線性限制，使得神經網路能夠逼近任何複雜的連續函數，從而大幅提升模型的表達能力（Expressivity）。

  * **參考文獻（Reference）：**

      * **Cybenko, G. (1989).** "Approximation by superpositions of a sigmoidal function." *Mathematics of Control, Signals and Systems*, 2(4), 303-314.
          * 這篇論文證明了具有 Sigmoid 激活函數的單隱藏層神經網路可以逼近任意連續函數。
          * [PDF 連結 (Semantic Scholar)](https://www.semanticscholar.org/paper/Approximation-by-superpositions-of-a-sigmoidal-Cybenko/21e82ed12c620fba1f5ee42162962aae74a23510)
      * **Hornik, K. (1991).** "Approximation capabilities of multilayer feedforward networks." *Neural Networks*, 4(2), 251-257.
          * 進一步證明了多層前饋網路是通用近似器（Universal Approximators），關鍵在於激活函數的非線性，而非特定的函數形式。
          * [PDF 連結 (Semantic Scholar)](https://www.semanticscholar.org/paper/Approximation-capabilities-of-multilayer-networks-Hornik/d35f1e533b72370683d8fa2dabff5f0fc16490cc)

-----

### 問題 2：How can we determine the number of layers and the number of neurons in a neural network?

**（我們該如何決定神經網路的層數與神經元數量？）**

  * **AI 研究解答：**
    這個問題屬於 **「超參數最佳化」（Hyperparameter Optimization, HPO）** 或更進階的 **「神經架構搜尋」（Neural Architecture Search, NAS）** 領域。
    目前**沒有**一個簡單的公式可以直接算出「最佳層數」，但研究界已經發展出比「試誤法」（Trial-and-Error）更有效的方法：

    1.  **Grid Search / Random Search**：隨機搜尋已被證明比網格搜尋更有效率。
    2.  **Bayesian Optimization**：利用貝氏機率模型來預測哪組參數可能表現更好。
    3.  **Neural Architecture Search (NAS)**：使用強化學習或進化演算法自動設計網路結構。

  * **參考文獻（Reference）：**

      * **Bergstra, J., & Bengio, Y. (2012).** "Random Search for Hyper-Parameter Optimization." *Journal of Machine Learning Research*, 13, 281-305.
          * 這篇經典論文證明了在尋找最佳參數（如層數、神經元數）時，隨機搜尋（Random Search）在效率上優於網格搜尋（Grid Search）。
          * [PDF 連結 (JMLR)](https://jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf)
      * **Elsken, T., Metzen, J. H., & Hutter, F. (2019).** "Neural Architecture Search: A Survey." *Journal of Machine Learning Research*, 20(55), 1-21.
          * 這是一篇關於自動化設計神經網路（包含決定層數與寬度）的綜述論文，詳細介紹了 NAS 的各種方法。
          * [PDF 連結 (ResearchGate)](https://www.researchgate.net/publication/327068329_Neural_Architecture_Search_A_Survey)

-----

### 問題 3：Why do we train the data in batches, how does this affect the training results?

**（為什麼我們要分批訓練數據，這如何影響訓練結果？）**

  * **AI 研究解答：**
    這涉及到 **Stochastic Gradient Descent (SGD)** 與 **Batch Gradient Descent** 的權衡。

    1.  **計算效率**：全量數據（Full Batch）記憶體放不下，單筆數據（Stochastic）又無法利用 GPU 平行運算優勢。Mini-batch 是兩者的最佳平衡點。
    2.  **泛化能力（Generalization）**：研究顯示，較小的 Batch Size 會在梯度估計中引入「雜訊」，這有助於模型跳出「銳利的局部最小值」（Sharp Minima），找到更平坦、泛化能力更好的極小值（Flat Minima）。

  * **參考文獻（Reference）：**

      * **Keskar, N. S., et al. (2017).** "On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima." *ICLR 2017*.
          * 這篇論文詳細探討了 Batch Size 對訓練結果的影響，指出大 Batch 訓練往往導致模型收斂到「尖銳」的極小值，從而導致測試集表現（泛化能力）變差。
          * [PDF 連結 (arXiv)](https://arxiv.org/pdf/1609.04836)
      * **Masters, D., & Luschi, C. (2018).** "Revisiting Small Batch Training for Deep Neural Networks." *arXiv preprint arXiv:1804.07612*.
          * 這篇研究進一步證實了小 Batch Size 在訓練穩定性和最終性能上的優勢。
          * [PDF 連結 (arXiv)](https://www.google.com/search?q=https://arxiv.org/pdf/1804.07612)

-----

# Week 2

### 問題：How to solve the problem of vanishing or exploding gradients?
**（如何解決梯度消失或梯度爆炸的問題？）**

* **問題描述**：
    在反向傳播（Backpropagation）中，如果每一層的梯度都小於 1（或大於 1），經過多層連乘後，傳遞到前面層的梯度會趨近於 0（梯度消失）或無限大（梯度爆炸）。這會導致模型無法學習或參數更新過大而發散。

* **AI 研究解答與文獻：**

    針對這個問題，學術界主要從以下四個方向提出了經典解決方案：

    1.  **權重初始化 (Weight Initialization)**：
        * **Xavier Initialization (Glorot Initialization)**：Glorot 與 Bengio 在 2010 年指出，使用標準梯度下降訓練深度網路之所以困難，部分原因在於激活函數的飽和。他們提出了一種標準化的初始化方法（Xavier init），確保每一層的輸出變異數（Variance）與輸入變異數一致，從而避免梯度在傳遞過程中消失或爆炸。
        * **He Initialization**：對於使用 ReLU 激活函數的網路，He 等人在 2015 年提出了針對性的初始化方法（He init）。他們證明了在極深網路（如 ResNet）中，這種初始化能更有效地保持信號強度，使模型能從頭開始訓練收斂。

    2.  **激活函數 (Activation Functions)**：
        * **ReLU (Rectified Linear Unit)**：相較於 Sigmoid 或 Tanh 函數在兩端會飽和（導數趨近 0），ReLU 在正區間的導數恆為 1，這解決了梯度消失的問題。He 等人的研究顯示，使用 PReLU（Parametric ReLU）等變體能進一步提升模型在 ImageNet 上的表現，甚至超越人類水平。

    3.  **標準化技術 (Normalization)**：
        * **Batch Normalization (BN)**：Ioffe 和 Szegedy 在 2015 年提出，訓練深層網路時，每層輸入的分佈會不斷變化（Internal Covariate Shift）。透過在每一層強制進行標準化（Mean=0, Variance=1），BN 不僅能防止梯度消失，還允許使用更高的學習率，並具有正則化效果，大幅加速了訓練過程。

    4.  **梯度裁剪 (Gradient Clipping)**：
        * 主要針對**梯度爆炸**問題。Pascanu 等人在 2013 年研究循環神經網路（RNN）的訓練困難時，提出了「梯度範數裁剪」（Gradient Norm Clipping）。當梯度的範數超過設定閾值時，直接將其縮放，這是一種簡單且有效的防止梯度爆炸策略。

---

### 參考文獻 (References)

1.  **Glorot, X., & Bengio, Y. (2010).** Understanding the difficulty of training deep feedforward neural networks. *Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics* (AISTATS), 9, 249-256.
    * [PDF 連結 (JMLR)](https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf)
2.  **He, K., Zhang, X., Ren, S., & Sun, J. (2015).** Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification. *ICCV 2015*.
    * [PDF 連結 (arXiv)](https://arxiv.org/abs/1502.01852)
3.  **Ioffe, S., & Szegedy, C. (2015).** Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift. *ICML 2015*.
    * [PDF 連結 (arXiv)](https://arxiv.org/abs/1502.03167)
4.  **Pascanu, R., Mikolov, T., & Bengio, Y. (2013).** On the difficulty of training Recurrent Neural Networks. *ICML 2013*.
    * [PDF 連結 (arXiv)](https://arxiv.org/abs/1211.5063)

---

# Week 3

### 問題 1：Why is Tanh less popular than ReLU or Sigmoid despite its theoretical advantages in approximating analytic functions?
**（為什麼 Tanh 在逼近解析函數上有理論優勢，卻不如 ReLU 流行？ Sigmoid 是否真的比 Tanh 流行？）**

* **AI 研究解答：**
    雖然 Tanh 函數是零中心化的（Zero-centered），在優化過程中比 Sigmoid 更有效率（Sigmoid 的輸出恆正會導致梯度更新方向呈鋸齒狀），但它仍然面臨嚴重的 **梯度消失問題（Vanishing Gradient Problem）**。
    1.  **梯度消失**：Tanh 在 $|x| > 2$ 時導數迅速趨近於 0，導致深層網絡無法有效傳遞梯度。這使得訓練深度網絡變得非常困難。
    2.  **計算成本**：Tanh 涉及指數運算，計算開銷比 ReLU 的簡單閾值運算（`max(0, x)`）大得多。
    3.  **ReLU 的稀疏性**：ReLU 會使一部分神經元輸出為 0，造成稀疏激發（Sparse Activation），這在生物學上更合理，且能減少參數間的耦合，提升泛化能力。
    4.  **Sigmoid 的現狀**：事實上，在現代深度學習中，Sigmoid 也**不再**流行於隱藏層，僅主要用於二元分類的輸出層。主流選擇早已轉向 ReLU 及其變體（Leaky ReLU, GeLU）。

* **參考文獻（Reference）：**
    * **LeCun, Y., Bottou, L., Orr, G. B., & Müller, K. R. (2012).** "Efficient BackProp." *Neural Networks: Tricks of the Trade*, 9-50.
        * 解釋了為何 Zero-centered 的 Tanh 比 Sigmoid 收斂更快，但也指出了飽和區間的梯度問題。
        * [PDF 連結 (Springer)](https://link.springer.com/chapter/10.1007/978-3-642-35289-8_3)
    * **Nair, V., & Hinton, G. E. (2010).** "Rectified Linear Units Improve Restricted Boltzmann Machines." *ICML 2010*.
        * 正式引入 ReLU，證明其在訓練深層網絡時比 Sigmoid/Tanh 更有效，解決了梯度消失問題。
        * [PDF 連結 (Omnipress)](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)
    * **Glorot, X., Bordes, A., & Bengio, Y. (2011).** "Deep Sparse Rectifier Neural Networks." *AISTATS 2011*.
        * 進一步探討了 ReLU 帶來的稀疏性優勢，以及其在計算效率上的巨大提升。
        * [PDF 連結 (JMLR)](http://proceedings.mlr.press/v15/glorot11a/glorot11a.pdf)

---

### 問題 2：Why do theoretical bounds (like Lemma 3.1) often fail in practice, requiring much larger networks? Is math useless in ML?
**（為什麼理論上的神經元數量在實作中不準？數學工具在機器學習中真的無用嗎？）**

* **AI 研究解答：**
    這是一個非常深刻的問題，觸及了 **「表達能力（Expressivity）」** 與 **「可優化性（Optimizability）」** 之間的巨大鴻溝。
    1.  **存在性 vs. 可構造性**：Lemma 3.1 等近似理論（Approximation Theory）證明了「存在」一組參數能達到 $\epsilon$ 誤差。但它**沒有告訴你如何透過梯度下降（Gradient Descent）找到這組參數**。
    2.  **損失曲面的複雜性**：當網路寬度剛好符合理論下界時，損失函數的曲面（Loss Landscape）通常非常崎嶇，充滿了局部極小值（Local Minima）或鞍點（Saddle Points），導致優化算法卡住。
    3.  **過參數化（Over-parameterization）的優勢**：近年來的研究（如 Neural Tangent Kernel, NTK）發現，當神經元數量遠大於理論需求時，損失曲面會變得更加平滑與凸化（Convex-like）。這使得梯度下降能輕易找到全域最優解（Global Minima）。雖然參數變多，但在適當的正則化下，模型反而具有更好的泛化能力（Double Descent 現象）。
    4.  **數學的價值**：數學並非無用，而是目前的數學工具（如經典 VC 維度理論）可能落後於深度學習的實驗進展。新的數學框架（如 NTK、Mean Field Theory）正在解釋「為什麼過參數化有效」，這正是數學指導實作的證明。

* **參考文獻（Reference）：**
    * **Li, H., et al. (2018).** "Visualizing the Loss Landscape of Neural Nets." *NeurIPS 2018*.
        * 透過可視化技術展示了深層網絡的損失曲面，並指出 Skip Connection 和寬度如何平滑曲面，使訓練變容易。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/1712.09913)
    * **Du, S. S., et al. (2019).** "Gradient Descent Finds Global Minima of Deep Neural Networks." *ICML 2019*.
        * 數學上證明了對於過參數化（Over-parameterized）的網路，梯度下降能在多項式時間內收斂到全域最優解。這解釋了為什麼實作中需要比理論下界更多的神經元。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/1811.03804)
    * **Belkin, M., et al. (2019).** "Reconciling modern machine-learning practice and the classical bias-variance trade-off." *PNAS*.
        * 提出了 "Double Descent" 曲線，解釋了為何在超過插值閾值（Interpolation Threshold）後，繼續增加參數反而能降低測試誤差，挑戰了傳統的統計學觀點。
        * [PDF 連結 (PNAS)](https://www.pnas.org/content/116/32/15849)

---

# Week 4

### 問題：想知道和 MSELoss 比起來，BCELoss 在二分類上的優勢是？

（在本次的二分類問題中，我分別嘗試了 OneHot + MSELoss 和講義中提到的 BCELoss ，並給予它們完全相同的神經網絡數值。結果發現，它們的結果並沒有多大區別，無論是誤差曲線還是成果圖，都長得差不多。唯一的不同就是 BCELoss 的訓練時間更長。）

  * **AI 研究解答：**
    儘管在簡單或特定的實驗設置下（如你的觀察），MSE 與 BCE 可能表現相似，但在理論與複雜場景下，BCE（Binary Cross Entropy）對於分類任務有著數學上的絕對優勢，主要體現在**梯度消失**與**機率解釋**兩個方面：

    1.  **解決梯度消失問題 (Vanishing Gradient Problem)**：

          * 當你使用 Sigmoid 激活函數搭配 MSE Loss 時，梯度的計算會包含 $\sigma'(z)$ 這一項（即 Sigmoid 的導數）。當預測值 $\hat{y}$ 接近 0 或 1 時（處於 Sigmoid 的飽和區）， $\sigma'(z)$ 會趨近於 0。這意味著即使模型的預測錯得離譜（例如標籤是 1，預測是 0.0001），梯度也會因為 $\sigma'(z) \approx 0$ 而變得極小，導致模型「學不動」。
          * 相反，**BCE Loss** 與 Sigmoid 的組合在數學推導上會產生一個完美的對消效果。其梯度形式簡化為 $(\hat{y} - y)$ ，不再包含 $\sigma'(z)$ 項。這保證了當預測誤差越大時，梯度越大，模型修正得越快。

    2.  **最大概似估計 (Maximum Likelihood Estimation, MLE)**：

          * 從統計學角度來看，分類問題的本質是預測機率。BCE Loss 等價於對 Bernoulli 分佈進行最大概似估計（MLE）。這使得 BCE 在優化機率模型時具有統計學上的一致性與高效性，而 MSE 則是假設誤差服從高斯分佈（Gaussian Distribution），這通常適用於回歸問題而非分類問題。

    3.  **優化曲面 (Optimization Landscape)**：

          * 研究指出，在分類問題上，MSE 容易導致損失函數曲面出現許多平坦區域（Plateaus）和非凸性（Non-convexity），這會阻礙梯度下降法的收斂。而 BCE 產生的曲面通常更陡峭，有助於加速收斂（儘管單次運算因為涉及 `log` 可能稍慢，但通常需要的 Epochs 更少）。

  * **參考文獻 (Reference)：**

      * **Golik, P., Doetsch, P., & Ney, H. (2013).** "Cross-Entropy vs. Squared Error Training: a Theoretical and Experimental Comparison." *Interspeech*.
          * 這篇論文詳細比較了 CE 與 MSE，證明了 CE 在大多數情況下收斂速度更快，且更能避免梯度消失問題，特別是在初始化不佳的情況下。
          * [PDF 連結 (RWTH Aachen University)](https://www.google.com/search?q=https://www-i6.informatik.rwth-aachen.de/publications/download/855/Golik-Interspeech-2013.pdf)
      * **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. (Chapter 6.2.2.3).
          * 深度學習聖經中明確指出：任何時候當我們設計神經網絡輸出機率時，使用 Cross-Entropy 總是優於 MSE，因為 MSE 會在飽和區導致學習極度緩慢。
          * [線上閱讀連結 (DeepLearning.org)](https://www.deeplearningbook.org/contents/mlp.html)
      * **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. (Chapter 4.3.2).
          * 從機率生成模型的角度解釋了為什麼 Cross-Entropy 是分類問題的自然選擇（Canonical Link Function）。
          * [PDF 連結 (Microsoft Research)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

---

# Week 5

### 問題 1：How does MLE change if the data are assumed independent but not identically distributed?

**（如果數據假設是獨立但非同分佈（Independent Non-Identically Distributed, INID），MLE 會有什麼變化？）**

  * **AI 研究解答：**
    在 I.I.D. 假設下，似然函數是相同機率密度函數的連乘積。若放寬為 INID（即數據 $x_i$ 來自不同的分佈 $P_i(x_i | \theta)$，但仍共享某些參數 $\theta$ ），MLE 的目標函數變為最大化各個不同分佈的對數似然之和：
    $$
    \ell(\theta) = \sum_{i=1}^n \log P_i(x_i | \theta)
    $$

    這在 **回歸分析**（每個 $y_i$ 的均值取決於 $x_i$ ）或 **異質變異數模型**（Heteroscedasticity，每個樣本變異數不同）中非常常見。

      * **理論性質變化**：雖然經典的大數法則（LLN）和中心極限定理（CLT）不能直接套用，但在滿足 **Lindeberg 條件** 或 **Lyapunov 條件** 下，INID 數據的 MLE 估計量仍然具有一致性（Consistency）和漸近常態性（Asymptotic Normality）。
      * **廣義線性模型 (GLM)**：這是處理 INID 數據最成熟的框架之一，其中每個樣本的分佈參數隨輸入變量而變。

  * **參考文獻 (Reference)：**

      * **White, H. (1980).** "A Heteroskedasticity-Consistent Covariance Matrix Estimator and a Direct Test for Heteroskedasticity." *Econometrica*.
          * 經典計量經濟學論文，解決了當數據獨立但變異數不同（非同分佈）時，如何進行正確的統計推斷。
          * [可疑链接已删除]
      * **Fahrmeir, L., & Kaufmann, H. (1985).** "Consistency and asymptotic normality of the maximum likelihood estimator in generalized linear models." *The Annals of Statistics*.
          * 證明了在 GLM 框架下（典型的 INID 情況），MLE 仍然具有良好的漸近性質。
          * [PDF 連結 (Project Euclid)](https://projecteuclid.org/euclid.aos/1176346597)

-----

### 問題 2：How do we compute MLE when some data points are missing?

**（當數據有缺失時，我們該如何計算 MLE？）**

  * **AI 研究解答：**
    處理缺失數據的 MLE 計算取決於缺失機制（Missing Mechanism）：

    1.  **MCAR (Missing Completely At Random)** 或 **MAR (Missing At Random)**：

          * 最標準的解法是 **EM 演算法 (Expectation-Maximization Algorithm)**。
          * **E-step**：利用現有的參數估計值，計算完整數據對數似然的期望值（填補缺失值的隱含分佈）。
          * **M-step**：最大化該期望值以更新參數。
          * 這個過程保證似然函數值單調遞增，直到收斂到局部最優解。

    2.  **MNAR (Missing Not At Random)**：

          * 缺失本身與數據值有關（例如：高收入者不願透露收入）。此時直接用 EM 會產生偏差，必須建立 **聯合模型** 同時對「數據生成過程」和「缺失機制」進行建模。

  * **參考文獻 (Reference)：**

      * **Dempster, A. P., Laird, N. M., & Rubin, D. B. (1977).** "Maximum Likelihood from Incomplete Data via the EM Algorithm." *Journal of the Royal Statistical Society*.
          * 這是提出 EM 演算法的開創性論文，是處理缺失數據 MLE 的黃金標準。
          * [可疑链接已删除]
      * **Little, R. J. A., & Rubin, D. B. (2019).** *Statistical Analysis with Missing Data*. Wiley.
          * 該領域的權威教科書，詳細定義了 MCAR, MAR, MNAR 並提供了對應的 MLE 計算方法。
          * [書籍連結 (Wiley)](https://www.google.com/search?q=https://www.wiley.com/en-us/Statistical%2BAnalysis%2Bwith%2BMissing%2BData%252C%2B3rd%2BEdition-p-9780470526698)

---

# Week 6

### 問題 1：GDA 如何擴展到多類別情形 ($K>2$) ?

  * **AI 研究解答：**
    GDA (Gaussian Discriminant Analysis) 可以直接推廣到多類別問題。
    在二元分類中，我們使用伯努利分佈（Bernoulli）來建模類別先驗 $P(y)$ 。在 $K$ 類別情形下，我們先驗改為使用 **多項式分佈 (Multinomial Distribution)**：
    $$
    P(y=k) = \phi_k, \quad \sum_{k=1}^K \phi_k = 1
    $$

    對於給定的類別 $k$ ，特徵 $x$ 仍然服從多變量高斯分佈：
    $$
    P(x|y=k) = \mathcal{N}(x; \mu_k, \Sigma_k)
    $$

    預測時，根據貝氏定理選擇後驗機率最大的類別：
    $$
    \hat{y} = \arg\max_k P(y=k|x) = \arg\max_k \left( \log P(x|y=k) + \log P(y=k) \right)
    $$
    
    這就是標準的 LDA (若 $\Sigma_k$ 相同) 或 QDA (若 $\Sigma_k$ 不同) 的多類別形式。

  * **參考文獻 (Reference)：**

      * **Murphy, K. P. (2012).** *Machine Learning: A Probabilistic Perspective*. MIT Press. (Chapter 4.2).
          * 詳細推導了 Generative Classifiers (如 GDA) 如何從二元擴展到多類別，並指出其決策邊界由 Log-likelihood ratio 決定。
          * [書籍連結 (MIT Press)](https://mitpress.mit.edu/9780262018029/machine-learning/)
      * **Hastie, T., Tibshirani, R., & Friedman, J. (2009).** *The Elements of Statistical Learning*. Springer. (Chapter 4.3).
          * 經典教科書，闡述了 LDA/QDA 在多類別情形下的判別函數（Discriminant Functions）形式。
          * [PDF 連結 (Stanford)](https://hastie.su.domains/ElemStatLearn/)

-----

### 問題 2：如果輸入需要標準化，應該在何時進行（訓練前，擬合時或預測時）？標準化會如何影響 $\mu, \Sigma$ 的估計與決策邊界？

  * **AI 研究解答：**

      * **何時進行**：標準化（Standardization, Z-score）應在**訓練前**對訓練集進行，並記錄下訓練集的 mean 和 std，然後用這些參數去轉換測試集/預測數據。
      * **對參數的影響**：
          * $\mu'$ ：變成原均值的線性變換版本（平移並縮放）。
          * $\Sigma'$ ：變成經過縮放的協方差矩陣（ $\Sigma' = D^{-1}\Sigma D^{-1}$ ，其中 $D$ 是標準差對角矩陣）。
      * **對決策邊界的影響**：
          * **理論上不變**：GDA/LDA 對於非奇異線性變換（Non-singular Linear Transformation）是**不變的 (Invariant)**。這意味著，無論你是否標準化數據，最終分類結果的預測（ $\arg\max$ ）是完全相同的。馬氏距離（Mahalanobis Distance）會自動補償特徵的尺度差異。
          * **實務上推薦**：雖然理論解不變，但標準化有助於數值穩定性（Numerical Stability），避免矩陣求逆時因數值差異過大而產生誤差。

  * **參考文獻 (Reference)：**

      * **Duda, R. O., Hart, P. E., & Stork, D. G. (2000).** *Pattern Classification*. Wiley.
          * 書中證明了 Fisher Linear Discriminant 對於特徵空間的線性變換（包括標準化）具有不變性。
          * [書籍連結 (Wiley)](https://www.wiley.com/en-us/Pattern+Classification%2C+2nd+Edition-p-9780471056690)
      * **Bishop, C. M. (2006).** *Pattern Recognition and Machine Learning*. Springer. (Chapter 4.1.4).
          * 討論了 Fisher 判別準則的性質，確認特徵縮放不會改變投影方向的相對關係。
          * [PDF 連結 (Microsoft)](https://www.microsoft.com/en-us/research/uploads/prod/2006/01/Bishop-Pattern-Recognition-and-Machine-Learning-2006.pdf)

-----

### 問題 3：對於那些邊界更加不規則的資料（比如，一個星形），該如何擴展 GDA 使其可用？

  * **AI 研究解答：**
    標準 GDA 假設數據呈現單峰高斯分佈（橢圓形簇），無法處理星形或環形等非凸（Non-convex）分佈。擴展方法主要有兩種：

    1.  **Gaussian Mixture Models (GMM) / Mixture Discriminant Analysis (MDA)**：
          * 放棄「每個類別只有一個高斯分佈」的假設。改為假設每個類別由**多個**高斯分佈組成（例如，星形的每個角是一個高斯子簇）。這使得模型能擬合任意形狀的複雜邊界。
    2.  **Kernel GDA (Kernel Discriminant Analysis, KDA)**：
          * 利用 **核技巧 (Kernel Trick)** 將數據映射到高維空間，使得在原空間中線性不可分（如星形中心與外圍）的數據在高維空間變得線性可分，然後在高維空間執行 GDA/LDA。

  * **參考文獻 (Reference)：**

      * **Hastie, T., & Tibshirani, R. (1996).** "Discriminant Analysis by Gaussian Mixtures." *Journal of the Royal Statistical Society*.
          * 提出了 Mixture Discriminant Analysis (MDA)，專門解決單一高斯無法擬合複雜類別分佈的問題。
          * [可疑链接已删除]
      * **Baudat, G., & Anouar, F. (2000).** "Generalized Discriminant Analysis using a Kernel Approach." *Neural Computation*.
          * 詳細介紹了如何將 GDA/LDA 核化（Kernelized），使其能處理高度非線性的分類邊界（如螺旋、星形數據）。
          * [PDF 連結 (MIT Press)](https://www.google.com/search?q=https://direct.mit.edu/neco/article/12/10/2385/6461/Generalized-Discriminant-Analysis-Using-a-Kernel)

---

# Week 7

### 問題 1：為什麼加噪聲要用高斯噪聲，用其他的會有什麼影響？（如：uniform）

* **AI 研究解答：**
    選擇高斯噪聲（Gaussian Noise）並非隨意，而是基於 **隨機微分方程 (SDE)** 與 **中心極限定理** 的數學特性：
    1.  **封閉形式的轉移機率 (Closed-form Transition Kernel)**：
        * 高斯分佈具有「再生性」（Reproductive Property）。若兩個變數服從高斯分佈，它們的和仍然是高斯分佈。這使得我們可以從 $x_0$ 直接跳到任意步數 $x_t$ （ $q(x_t|x_0)$ 仍是高斯），而無需一步步計算。若使用 Uniform Noise，多次疊加後分佈形狀會改變（趨向高斯但過程複雜），無法擁有這種解析解優勢。
    2.  **與物理擴散過程的聯繫**：
        * 擴散模型本質上是在模擬布朗運動（Brownian Motion）。在連續時間極限下，擴散過程由 SDE 描述： $dx = f(x, t)dt + g(t)dw$ ，其中 $dw$ 是維納過程（Wiener Process），其增量嚴格服從高斯分佈。
    3.  **其他噪聲的研究**：
        * 近期也有研究探索非高斯擴散（例如基於 **Levy Flights** 的重尾分佈噪聲），試圖加速採樣或處理多模態分佈，但數學推導遠比高斯複雜，且目前尚未成為主流。

* **參考文獻 (Reference)：**
    * **Sohl-Dickstein, J., et al. (2015).** "Deep Unsupervised Learning using Nonequilibrium Thermodynamics." *ICML 2015*.
        * 擴散模型的奠基之作，利用擴散過程（高斯變換）將複雜數據分佈轉化為簡單高斯分佈，並學習逆過程。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/1503.03585)
    * **De Bortoli, V., et al. (2022).** "Riemannian Score-Based Generative Modelling." *NeurIPS 2022*.
        * 探討了在流形上或其他幾何結構下的擴散，雖然可以是非歐氏幾何，但局部噪聲通常仍近似高斯以利用其數學便利性。

---

### 問題 2：AI 圖片生成初期常有邏輯失誤（如手指沾黏），除了增加去噪輪數，工程師們是如何改進模型來解決這個問題的？

* **AI 研究解答：**
    「手指沾黏」或「邏輯錯誤」通常不是因為去噪步數不夠，而是模型對 **高頻細節** 的理解不足或 **語意對齊（Semantic Alignment）** 較弱。改進並非單靠增加步數，而是透過架構與訓練策略的革新：
    1.  **潛在空間擴散 (Latent Diffusion Models, LDM / Stable Diffusion)**：
        * 不在像素空間（Pixel Space）而在壓縮的潛在空間（Latent Space）進行擴散。這不僅大幅降低運算量，還能讓模型更專注於圖像的「語意結構」而非微小的像素雜訊，提升了整體結構的合理性。
    2.  **強化文本編碼器 (Better Text Encoders)**：
        * Google 的 **Imagen** 研究指出，使用更強大的語言模型（如 T5-XXL）來提取文本特徵，比單純擴大擴散模型本身更能提升生成品質（尤其是計數與邏輯理解能力）。
    3.  **無分類器引導 (Classifier-Free Guidance, CFG)**：
        * 這是提升圖像與提示詞（Prompt）一致性的關鍵技術。透過在訓練時隨機丟棄文本條件，並在採樣時放大「有條件」與「無條件」預測的差異，強制模型更嚴格地遵循邏輯描述（如「五根手指」）。
    4.  **空間控制與微調 (ControlNet & Adapters)**：
        * 引入額外的空間條件（如骨架圖、邊緣圖）來顯式地告訴模型手部結構，這是解決肢體崩壞最直接有效的方法。

* **參考文獻 (Reference)：**
    * **Rombach, R., et al. (2022).** "High-Resolution Image Synthesis with Latent Diffusion Models." *CVPR 2022*.
        * Stable Diffusion 的原論文，提出了在 Latent Space 進行訓練以提升效率與結構品質。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/2112.10752)
    * **Ho, J., & Salimans, T. (2022).** "Classifier-Free Diffusion Guidance." *NeurIPS 2021 Workshop*.
        * 提出了 CFG 技術，這是現代擴散模型能生成符合邏輯描述圖片的核心技巧。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/2207.12598)
    * **Saharia, C., et al. (2022).** "Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding." *NeurIPS 2022*.
        * Google Imagen 論文，證明了「語言模型的大小」對生成圖片的邏輯性（如數對物體數量）至關重要。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/2205.11487)

---

### 問題 3：關於 Hutchinson Trace Estimator：為什麼 $\mathbb{E}[v^T A v]$ (右式) 比 $\text{tr}(A)$ (左式) 更好算？左式不是 $O(n)$ 而右式是 $O(n^2)$ 嗎？

* **AI 研究解答：**
    你的直覺在「矩陣 $A$ 已經被完整計算出來」的情況下是正確的。但在 Implicit Score Matching (ISM) 的場景中，關鍵在於 **我們並不知道矩陣 $A$ 的具體數值**。
    * **背景**：這裡的 $A$ 是分數函數的 **雅可比矩陣 (Jacobian Matrix)** $\nabla_x S_\theta(x)$ 。對於一張 $64 \times 64 \times 3$ 的圖片，維度 $n \approx 12,288$ 。
    * **左式困境 ( $\text{tr}(A)$ )**：
        * 要計算對角線元素 $\frac{\partial S_i}{\partial x_i}$ ，我們必須對模型的每一個輸出維度分別進行反向傳播（Backpropagation）。這需要執行 $n$ 次反向傳播，計算成本是 $O(n)$ 次神經網絡推論，這在 $n$ 很大時是不可接受的。
    * **右式優勢 ( $\mathbb{E}[v^T A v]$ )**：
        * 這裡利用了 **自動微分 (Autodiff)** 的特性。我們不需要算出整個矩陣 $A$ 。
        * $v^T A v = v^T (\nabla_x S_\theta(x)) v$ 。
        * 我們可以先計算 **向量-雅可比積 (Vector-Jacobian Product, vJP)**： $u = v^T \nabla_x S_\theta(x)$ 。在現代深度學習框架（如 PyTorch, TensorFlow）中，計算 $v^T J$ 只需要 **一次** 反向傳播（成本約等於一次前向傳播），而不需要算出 $J$ 的每一個元素。
        * 因此，計算右式的成本大約是 $O(1)$ 次反向傳播（假設採樣次數固定），而不是左式的 $O(n)$ 次。這就是為什麼說右式在計算效率上具有壓倒性優勢。

* **參考文獻 (Reference)：**
    * **Hutchinson, M. F. (1989).** "A stochastic estimator of the trace of the influence of a data smoothing matrix." *Communications in Statistics-Simulation and Computation*.
        * 提出了利用隨機向量投影來估計矩陣跡（Trace）的方法。
        * [文章連結 (Taylor & Francis)](https://www.tandfonline.com/doi/abs/10.1080/03610919008812866)
    * **Song, Y., et al. (2019).** "Generative Modeling by Estimating Gradients of the Data Distribution." *NeurIPS 2019*.
        * 在 Sliced Score Matching 中應用了這個技巧，明確指出利用 vJP 可以避免計算高維雅可比矩陣，從而將計算複雜度與數據維度解耦。
        * [PDF 連結 (arXiv)](https://arxiv.org/abs/1907.05600)

---

# Week 8

### 問題 1：歐拉-丸山法 (Euler-Maruyama) 的收斂速度和精度如何？是否存在比它更精確、收斂更快的數值方法？

* **AI 研究解答：**
    * **Euler-Maruyama 的精度**：它是最簡單的 SDE 數值解法，類似於 ODE 中的 Euler 法。它具有 **0.5 階的強收斂 (Strong Convergence)** 和 **1.0 階的弱收斂 (Weak Convergence)**。這意味著當時間步長 $\Delta t \to 0$ 時，其平均路徑誤差以 $\sqrt{\Delta t}$ 的速度減小，這在許多高精度要求的應用中是不夠快的。
    * **更高級的方法**：
        * **Milstein Method**：它透過在泰勒展開中加入二階隨機項（伊藤修正項），達到了 **1.0 階的強收斂**。雖然精度提高，但它需要計算擴散項 $g(x,t)$ 的導數 $g'(x,t)$ ，這在高維情況下計算量較大。
        * **Runge-Kutta for SDEs (Strong Order 1.0/1.5)**：類似於 ODE 的 RK4，這類方法不需要計算導數，通過多步預測-校正來獲得高階收斂，適合不希望手動推導導數的場景。
        * **Exponential Integrators (e.g., DDIM)**：在擴散模型（Diffusion Models）的特定應用中，如果 SDE 是線性的或半線性的（如 OU 過程），使用指數積分器可以精確求解線性部分，從而允許極大的步長（如 DDIM 可在 50 步內生成高質量圖像，而 Euler-Maruyama 可能需要 1000 步）。

* **參考文獻 (Reference)：**
    * **Kloeden, P. E., & Platen, E. (1992).** *Numerical Solution of Stochastic Differential Equations*. Springer.
        * SDE 數值方法的聖經級教科書，詳細定義了強/弱收斂階數，並推導了 Milstein 和各類 Taylor 展開方法。
    * **Song, J., Meng, C., & Ermon, S. (2021).** "Denoising Diffusion Implicit Models." *ICLR 2021*.
        * 提出了 DDIM，這本質上是一種非馬可夫的採樣過程，也可視為對應 ODE 的高階數值解法，解決了擴散模型採樣慢的問題。

---

### 問題 2：既然 SSM (Sliced Score Matching) 可以讓模型去學習高維度的數據，那有沒有可能用 SSM 製作一個動畫生成模型？（二維畫面+時間軸）

* **AI 研究解答：**
    * **理論可行性**：完全可行。動畫可以被視為一個 3D 張量 $X \in \mathbb{R}^{T \times H \times W}$ （時間 $\times$ 高 $\times$ 寬）。SSM 的優勢在於它不依賴計算雅可比矩陣的 Trace，這使得它在處理超高維數據（如影片像素總數遠大於單張圖片）時，記憶體和計算效率仍具優勢。
    * **實際挑戰與解法**：
        * **維度爆炸**：雖然 SSM 解決了 Trace 計算，但高維空間的「稀疏性」使得模型難以捕捉長距離的時間依賴（Temporal Consistency）。
        * **Video Diffusion Models (VDM)**：目前的 SOTA 影片生成模型（如 Google 的 VideoPoet, OpenAI 的 Sora）雖然核心通常基於 Denoising Score Matching (DSM) 而非 SSM（因為 DSM 在擴散框架下更自然），但它們處理「時間軸」的邏輯是通用的：將 2D U-Net 擴展為 **3D U-Net**，或在 Spatial Attention 後串接 **Temporal Attention** 層。
        * **SSM 的特定應用**：SSM 更常被用於那些「無法輕易獲得條件機率 $p(x_t|x_0)$ 」的能量模型（Energy-Based Models, EBM）或非標準擴散過程。如果動畫生成被建模為一個連續軌跡的優化問題，SSM 是一個強有力的候選方案。

* **參考文獻 (Reference)：**
    * **Song, Y., et al. (2021).** "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.
        * 這篇論文統一了 Score Matching 與 Diffusion Models，並展示了如何用 SDE 求解器生成連續訊號（理論上包含影片）。
    * **Ho, J., et al. (2022).** "Video Diffusion Models." *NeurIPS 2022*.
        * Google Research 團隊將擴散模型擴展到影片生成的奠基之作，展示了如何處理時間維度的架構設計。
    * **Harvey, W., et al. (2022).** "Flexible Diffusion Modeling of Long Videos." *NeurIPS 2022*.
        * 探討了如何生成長影片，這涉及了在時間軸上的分段生成與一致性維護，雖然主要用 DSM，但其對高維時空數據的處理邏輯與 SSM 的目標一致。

---

# Week 10

### 問題 1：在實際複雜系統（例如氣候或生物網路）中，使用機率流 ODE 作為降維或生成模型是否具有可擴展性？其計算與統計瓶頸在什麼規模開始變得不可行？

* **AI 研究解答：**
    Probability Flow ODE (PF-ODE) 將隨機的 SDE 軌跡轉化為確定性的 ODE 軌跡，理論上這極大簡化了採樣過程。但在處理大規模複雜系統時，它面臨著以下關鍵瓶頸：

    1.  **剛性 (Stiffness) 問題**：
        * 這是 PF-ODE 最主要的計算瓶頸。氣候或生物網絡等複雜系統通常包含多個時間尺度的動力學（例如：快速的化學反應 vs 慢速的生態演化）。這會導致對應的 ODE 變得極度「剛性」（Stiff）。數值求解器為了保持穩定，必須採用極小的步長，導致採樣時間變得無法接受（例如生成一張高解析度氣候圖可能需要數千次 ODE 步進）。
        * 雖然有一些針對性的加速器（如 DPM-Solver），但當數據維度極高且非線性極強時，剛性問題依然顯著。

    2.  **高維分數估計誤差 (Score Estimation Error)**：
        * PF-ODE 的軌跡完全依賴於分數函數 $\nabla_x \log p(x)$ 的準確性。在高維空間（如氣候模擬中的 $10^6$ 維度狀態向量），分數估計的微小誤差會在 ODE 積分過程中被累積放大，導致軌跡偏離真實流形（Manifold）。這在生成模型中表現為「幻覺」或不切實際的樣本。

    3.  **統計瓶頸 (Statistical Bottleneck)**：
        * 降維的可擴展性受限於 **流形假設 (Manifold Hypothesis)** 的有效性。氣候系統的數據雖然維度高，但往往位於低維流形上。然而，PF-ODE 試圖在整個環境空間（Ambient Space）中建模機率流，這在數據稀疏的高維區域效率極低。

* **參考文獻 (Reference)：**
    * **Lu, C., et al. (2022).** "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Models." *NeurIPS 2022*.
        * 深入分析了擴散模型對應的 ODE 的半線性結構與剛性問題，並提出了專用的高階求解器來緩解計算瓶頸。
    * **Song, Y., et al. (2021).** "Score-Based Generative Modeling through Stochastic Differential Equations." *ICLR 2021*.
        * 雖然證明了 PF-ODE 的存在性，但也指出了在極高維度下，數值積分誤差與分數估計誤差的耦合是主要限制。
    * **Karras, T., et al. (2022).** "Elucidating the Design Space of Diffusion-Based Generative Models." *NeurIPS 2022*.
        * 詳細探討了 ODE 軌跡的曲率（Curvature）與採樣步數的關係，指出複雜數據分佈會導致 ODE 軌跡高度彎曲，從而限制了大規模應用的效率。
