# Week 7 Assignment


## Problem 1

> Explain the concept of score matching and describe how it is used in score-based (diffusion) generative models.

### 核心概念

Score Matching（分數匹配）是一种用來數據分佈梯度（即score function）的方法

定義score function：

$$
S(x) = \nabla_x \log p(x)
$$

直觀來說，分數 $S(x)$ 是一個向量，它指向的位置是機率密度 $p(x)$ 增加最快的方向

### 目標

給定原數據概率分佈 $p_{\text{data}}(x)$ ，希望訓練一個模型 $S(x; \theta)$ ，使得模型接近原數據的分數，也就是

$$
S(x; \theta) \approx \nabla_x \log p_{\text{data}}(x)
$$

### 損失函數

#### 方案一：Explicit score matching (ESM)

$$
L_{ESM}(\theta) = \mathbb{E}_{x\sim p(x)}\|S(x; \theta)-\nabla_x\log p_{\text{data}}(x)\|^2
$$

這個很直觀，但在實際問題中，原數據的分佈，也就是分數函數 $\nabla_x\log p_{\text{data}}(x)$ 通常是未知的

#### 方案二：Implicit score matching (ISM)

$$
L_{ISM}(\theta) = \mathbb{E}_{x\sim p(x)}\left[\|S(x; \theta)\|^2+2\nabla_x\cdot S(x; \theta)\right]
$$

此式由ESMLoss推導而來，避開了對未知數據 $\nabla_x\log p_{\text{data}}(x)$ 的依賴，只和我們的模型 $s_\theta(x)$ 以及從真實數據中抽樣的點 $x$ 有關

### 應用：扩散模型（diffusion models）

直接在原始數據上學習分數函數很困難，因為真實數據的分佈可能非常複雜（例如，圖像空間中可能有很多分離的「山峰」） \
擴散模型提出了一個解決方案：**先加噪，再學習去噪**

#### 第一步：前向傳播

我們不直接學習原始、乾淨數據 $p_0(x)$ 的分數，而是故意對其進行「污染」 \
我們從一個真實的數據樣本 $x_0$ 開始，在 $T$ 個時間步中，逐步地向它添加高斯噪聲，直到它最終變成一個正態分佈的隨機噪聲 $x_T \sim \mathcal{N}(0, I)$

$$
x_0 \xrightarrow{\text{noise}} x_1 \xrightarrow{\text{noise}} \dots \xrightarrow{\text{noise}} x_T \sim \mathcal{N}(0, I)
$$

#### 第二部：反向傳播

我們從 $x_T$ 開始，然後逐步去除噪聲，最終恢復出一個清晰的數據樣本 $x_0$

$$
x_T \xrightarrow{\text{denoising}} x_{T-1} \xrightarrow{\text{denoising}} \dots \xrightarrow{\text{denoising}} x_0
$$

這個去噪的過程正是模型需要學習的部分

#### Denoising score matching (DSM)

符號
- $x_0$ : 原數據
- $p_0(x_0)$ : 原數據分佈
- $x$ : 噪聲數據（通過擾亂原數據得到）
- $p(x|x_0)$ : 條件（噪聲）數據分佈
- $p_{\sigma}(x)$ : （噪聲）數據分佈

通過一些計算，我們可以設計損失函數

$$
L_{DSM}(\theta) = \mathbb{E}_{x_0\sim p_0(x_0)}\mathbb{E}_{x|x_0\sim p(x|x_0)}\left[\|S_\sigma(x;\theta)-\nabla_{x}\log p(x|x_0)\|^2\right]
$$

相同地，這裡的 $\nabla_{x}\log p(x|x_0)$ 需要進行一些處理

關注單次的加噪過程

$$
x = x_0 + \sigma\epsilon, \quad \epsilon\sim \mathcal{N}(0, I)
$$

經過一些計算，我們會得到

$$
\nabla_x\log p(x|x_0)= -\frac{1}{\sigma^2} (x-x_0)=-\frac{1}{\sigma^2} \epsilon_\sigma
$$

最後我們會得到

$$
L_{DSM}(\theta) = \mathbb{E}_{x_0\sim p_0(x_0)}\mathbb{E}_{x|x_0\sim p_\sigma(x|x_0)}\left\|S_\sigma(x;\theta)+\frac{x-x_0}{\sigma^2} \right\|^2
$$

#### 結語

擴散模型是一種生成式模型 \
可以發現我們給原數據加噪聲，最後得到了一個完全混亂的數據 \
造成的結果就是，其實這些噪聲圖每張都長得差不多，我們分不清，模型也分不清 \
所以就算你最後給了模型一張真正的噪聲圖（隨機生成的數據），它也一樣能把數據給還原回來，這就是擴散模型的強大之處


## Problem 2

> Unanswered Questions

1. 為什麼加噪聲要用高斯噪聲，用其他的會有什麼影響？（如：uniform）
2. 圖片生成ai剛開始流行時，生成的圖片會有很多邏輯性失誤（如：人的手指糊在一起），ai不太能理解這些圖片中的細節含義。而且擴散模型除了增加去噪輪數似乎也沒多少改進空間了。但是，現在這些問題漸漸得到了改善，ai工程師們是如何改進模型來解決這個問題的呢？
3. 課堂上提到過：\
    為了在實際問題中的計算效率，我們會改寫 $\text{tr}(A) = \mathbb{E}_{v\sim p(v)}[v^TAv]$ ，來簡化程式的運算量。但我想了想，左式是對角線元素的和，複雜度是 $O(n)$ ；而右式要做一次矩陣向量乘法， 複雜度 $O(n^2)$ ，為什麼會說右邊比較好算呢？
