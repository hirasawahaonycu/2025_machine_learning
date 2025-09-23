# Problem 1


## 主旨

兩個引理旨在證明，單項式可以被一個淺層tanh網絡逼近，並計算它們的誤差估計和權重規模。 \
這將為之後證明tanh網絡可以逼近解析函數提供基礎。


## 先備知識

### Sobolev 空间 和 $W^{k, \infty}$-范數

> 给定一个区域 $\Omega \subseteq \mathbb{R}^{d}$，整数 $k \geq 0$ 以及 $1 \leq p \leq \infty$，Sobolev 空间定义为
> 
> $$
> W^{k,p}(\Omega) = \left\{ f \in L^p(\Omega) \mid D^\alpha f \in L^p(\Omega) \ \forall \alpha \in \mathbb{N}_0^d \ \text{with } |\alpha| \leq k \right\}.
> $$
> 
> 當 $p = \infty$ 時，定義
> 
> $$
> \parallel f \parallel_{W^{k,\infty}(\Omega)} 
> = \max_{0 \leq m \leq k} \max_{|\alpha| = m} \parallel D^\alpha f \parallel_{L^\infty(\Omega)}.
> $$


簡而言之，$W^{k, \infty}$ -范數是用來衡量函數和它的各階導數的最大值的。\
它要求從函數 $f$ 的前 $k$ 階導數和函數值本身中挑出最大值，即 $f, Df, D^2f, ..., D^kf$，例如：
- $W^{0, \infty}$ ：函數的最大值
- $W^{1, \infty}$ ：函數和其導數的最大值
- $W^{k, \infty}$ ：函數和其前 $k$ 階導數的最大值

在這篇論文里 $W^{k, \infty}$ 逼近的意思就是，在神經網絡和原函數的誤差計算中，不僅僅是函數值需要逼近，它們的前 $k$ 階導數也要逼近。

### 差分算子

令 $p \in \mathbb{N}$ 和 $f \in C^{p+2}$，有限差分算子定義為

$$
\delta^p_h[f] (x) = \sum_{i=0}^{p} (-1)^i \binom{p}{i} f\!\left(x+\Big(\tfrac{p}{2}-i\Big)h \right).
$$

它能用來近似函數的 $p$-階導數：

$$
\delta^p_h[f] (x) \rightarrow h^p \cdot f^{(p)}(x) \quad \text{ as } h \rightarrow 0
$$

在接下來的解析中，我們定義

$$
\hat{f}_{q, h}(x) := \frac{\delta^{q}_{hx}[\sigma] (0)}{\sigma^{(q)}(0) h^{q}}
$$


## Lemma 3.1

> 設 $k \in \mathbb{N}_0$，且 $s \in 2\mathbb{N}-1$（即 $s$ 為奇數）。  
> 則對任意 $\epsilon > 0$，存在一個淺層 tanh 神經網路
> 
> $$
> \Psi_{s,\epsilon} : [-M,M] \to \mathbb{R}^{\tfrac{s+1}{2}}
> $$
> 
> 其寬度為 $\tfrac{s+1}{2}$，使得
> 
> $$
> \max_{p \leq s,\ p \ \text{odd}} 
> \parallel f_p - (\Psi_{s,\epsilon})_{\tfrac{p+1}{2}} \parallel_{W^{k,\infty}} \leq \epsilon,
> $$
> 
> 其中 $f_p(x) = x^p$。  
> 
> 此外，該神經網路的權重大小滿足以下增長規律：
> 
> $$
> O\!\left(\epsilon^{-\tfrac{s}{2}} \, \big(2(s+2)\sqrt{2}M\big)^{s(s+3)}\right),
> $$
> 
> 當 $\epsilon \to 0$ 且 $s$ 很大時成立。

### 神經網絡

- 輸入一個 $x$
- 隱藏層 $\frac{s+1}{2}$ 個神經元
- 輸出 $\frac{s+1}{2}$ 個值，分別表示 $f(x), f(x^3), f(x^5), ..., f(x^s)$

### 解析

這個引理指出，只要 $s$ 是一個奇數，就一定能找到一個這樣的淺層tanh網絡，使得網絡的輸出和單項式 $f_p(x) := x^p$ 的 $W^{k, \infty}$ 誤差小於 $\epsilon$，而且 $\epsilon > 0$ 是任取的

再說得簡單點，在理論上，這個淺層tanh網絡可以逼近任意一個奇數階的單項式，而且是很好的逼近（它們的高階導數也能逼近）

當然， $\epsilon$ 取得越小，神經網絡的權重規模也就越大 \
引理也指出，這些權重規模是可以被限制住的，它大約為

$$
\text{權重大小} \sim O\!\left(\epsilon^{-\tfrac{s}{2}}\right)
$$

### 證明思路

觀察tanh的泰勒展開式

$$
\text{tanh}(x) = x - \frac{1}{3} x^3 + \frac{2}{15} x^5 - ...
$$

注意到，tanh的展開式包含了所有的奇數階單項式 \
於是，我們似乎可以用 tanh的導數+適當的歸一化 提取出個別單項式

而tanh的導數可以通過tanh的 放縮平移+線性組合 得到（用差分法實現），如下：

$$
h^p \cdot \text{tanh}^{(p)}(x)
\approx \delta^p_h[\text{tanh}] (x)
= \sum_{i=0}^{p} (-1)^i \binom{p}{i} \text{tanh}\!\left(x+\Big(\tfrac{p}{2}-i\Big)h \right).
$$

而且，因為tanh是奇函數，即 $\text{tanh}(-x) = -\text{tanh}(x)$，所以我們可以對稱地取代掉一半的隱藏層神經元 \
所以隱藏層只需要 $\frac{s+1}{2}$ 個神經元足矣

然後，我們定義了導數的歸一化

$$
\hat{f}_{q, h}(x) := \frac{\delta^{q}_{hx}[\sigma] (0)}{\sigma^{(q)}(0) h^{q}}
$$

任務就是證明

$$
\hat{f}_{q, h}(x) \approx f_p(x) := x^p
$$

嚴格來說

$$
\Big\parallel \hat{f}_{q, h}(x) - f_p(x) \Big\parallel_{W^{k,\infty}} \leq \epsilon
$$

這裡的證明就十分複雜了，略


## Lemma 3.2

> 設 $k \in \mathbb{N}_0$， $s \in 2\mathbb{N}-1$（即 $s$ 為奇數），且 $M > 0$。  
> 對任意 $\epsilon > 0$，存在一個淺層 tanh 神經網路  
> 
> $$
> \Psi_{s,\epsilon} : [-M,M] \to \mathbb{R}^s
> $$
> 
> 其寬度為 $\tfrac{3(s+1)}{2}$，使得
> 
> $$
> \max_{p \leq s} \big\parallel f_p - (\Psi_{s,\epsilon})_p \big\parallel_{W^{k,\infty}} \leq \epsilon,
> $$
> 
> 其中 $f_p(x) = x^p$。  
> 
> 此外，該神經網路的權重大小滿足以下增長規律：
> 
> $$
> O\!\left(\epsilon^{-\tfrac{s}{2}} \, \big(\sqrt{M(s+2)}\big)^{\tfrac{3s(s+3)}{2}}\right),
> $$
> 
> 當 $\epsilon \to 0$ 且 $s$ 很大時成立。

### 神經網絡

- 輸入一個 $x$
- 隱藏層 $\frac{3(s+1)}{2}$ 個神經元
- 輸出 $s$ 個值，分別表示 $f(x), f(x^2), f(x^3), ..., f(x^s)$

### 解析

引理指出，只要 $s$ 是一個奇數，就一定能找到一個這樣的淺層tanh網絡，使得網絡的輸出和單項式 $f_p(x)$ 的 $W^{k, \infty}$ 誤差小於 $\epsilon$，而且 $\epsilon > 0$ 是任取的。

與引理 3.1 類似的，權重規模有

$$
\text{權重大小} \sim O\!\left(\epsilon^{-\tfrac{s}{2}}\right)
$$

注意，這個引理和 3.1 的區別在於，3.1 只能列出奇數項，而這裡可以列出小於等於 $s$ 的所有項。也就是說，這個 3.2 是 3.1 的加強版。

### 證明思路

奇數項的情況已經在引理 3.1 全部解決了，現在只差偶數項 \
因為tanh的展開式只包含奇數項，所以我們無法用 3.1 的方法解出偶數項 \
為了解決這個問題，我們需要引用這樣一個公式：

對於任意 $n \in \mathbb{N}, \alpha > 0$ ，有

$$
y^{2n} = \frac{1}{2\alpha(2n+1)} \Bigg( (y+\alpha)^{2n+1} - (y-\alpha)^{2n+1} - 2 \sum_{k=0}^{n-1} \binom{2n+1}{2k} \alpha^{\,2(n-k)+1} \, y^{2k} \Bigg)
$$

這個公式告訴我們，偶數項可以通過「兩個奇數項的差」再減去「一些低階偶數項」得到 \
因此，我們可以遞歸地構造偶數項

然後我們也需要更多的隱藏層神經元

證明複雜，同略


# Problem 2

1. 這篇論文指出，tanh激活函數在對解析函數的逼近的問題上有十分優異的效果。但為什麼tanh卻不如其他，如ReLU，Sigmoid等激活函數來得更流行呢？

2. 在這個章節中，我們花了很大的力氣來計算證明，試圖為模型的選擇提供理論上的支持。但是，正如前面所說，這些證明出來的結果往往也真的只是「理論上」的，到了實際問題上，結果往往與理論結果大相徑庭。就拿以上解析的 Lemma-3.1 為例，雖然已經證明了存在隱藏層只需 $\frac{s+1}{2}$ 個神經元的理論解，但實作後就會發現只放 $\frac{s+1}{2}$ 根本不准，想要效果好還需要更多（好幾倍）的神經元。\
我的意思是，目前為止，我並沒有在機器學習這個領域上看到數學工具的強大
