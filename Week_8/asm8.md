# Week 8 Assignment


## Problem 1

> Show that the sliced score matching (SSM) loss can also be written as
> $$
> L_{SSM}=\mathbb{E}_{x\sim p(x)} \mathbb{E}_{v\sim p(v)} \left[\|v^TS(x;\theta)\|^2+2v^T\nabla_x (v^TS(x;\theta))\right]
> $$

從LSM開始，

$$
L_{ISM}(\theta) = \mathbb{E}_{x\sim p(x)}\left[\|S(x;\theta)\|^2 +2\nabla_x\cdot S(x;\theta)\right]
$$

令 $v$ 為一個隨機向量，且滿足 $v\sim N(0,I)$ ，則有

$$
\mathbb{E}[vv^T]=I
$$

### Lemma

$$
\text{tr}(A) = \mathbb{E}_{v\sim p(v)}[v^TAv]
$$

> *proof*
>
> $$
> \begin{align*}
>     \mathbb{E}_{v\sim p(v)}[v^TAv] &= \mathbb{E}_{v\sim p(v)}[\text{tr}(v^TAv)] \\
>     &=\mathbb{E}_{v\sim p(v)}[\text{tr}(Avv^T)]\\
>     &=\text{tr}(A\,\mathbb{E}_{v\sim p(v)}[vv^T])\\
>     &=\text{tr}(A)
> \end{align*}
> $$

我們分兩步來證明
- 第一步：證明 $\|S\|^2 = \mathbb{E}_{v\sim p(v)} \|v^TS\|^2$
- 第二步：證明 $\nabla_x\cdot S = \mathbb{E}_{v\sim p(v)}\left[v^T\nabla_x (v^TS) \right]$

### 第一步：證明 $\|S\|^2 = \mathbb{E}_{v\sim p(v)} \|v^TS\|^2$

首先，

$$
\|v^TS\|^2 = (v^TS)^2 = v^T(SS^T)v
$$

則由Lemma，可得

$$
\mathbb{E}_{v} \|v^TS\|^2 = \mathbb{E}_{v} (v^T(SS^T)v) = \text{tr}(SS^T) = \|S\|^2
$$

### 第二步：證明 $\nabla_x\cdot S = \mathbb{E}_{v\sim p(v)}\left[v^T\nabla_x (v^TS) \right]$

散度可以寫作 Jacobian 的 trace：

$$
\nabla_x \cdot S(x) = \text{tr}(\nabla_x S(x))
$$

由Lemma，可得

$$
\text{tr}(\nabla_x S(x)) = \mathbb{E}_v[v^T (\nabla_x S(x)) v]
$$

注意，我們可以證明

$$
v^T(\nabla_x S(x))v = v^T\nabla_x(v^T S(x))
$$

> *proof*
>
> 令 $S(x) = [S_1(x), S_2(x), ..., S_d(x)]^T$ ，則
>
> $$
> v^T S(x) = \sum_i v_i S_i(x)
> $$
>
> 對 $x$ 求梯度：
>
> $$
> \nabla_x(v^T S(x)) = \sum_i v_i \nabla_x S_i(x) = (\nabla_x S(x))^T v
> $$
>
> 再左乘 $v^T$ ，得到：
>
> $$
> v^T \nabla_x(v^T S(x)) = v^T (\nabla_x S(x)) v
> $$
>
> 因此兩者等價

繼續，總結以上，我們得到

$$
\nabla_x \cdot S(x) = \mathbb{E}_v \left[v^T \nabla_x(v^T S(x)) \right]
$$

### 合併得到SSM

根據上面兩個推導出的等式，代入LSM：

$$
\begin{align*}
    L_{ISM}(\theta)
    &= \mathbb{E}_{x\sim p(x)}\left[\|S(x;\theta)\|^2 +2\nabla_x\cdot S(x;\theta)\right] \\
    &= \mathbb{E}_{x\sim p(x)} \mathbb{E}_{v\sim p(v)} \left[\|v^TS(x;\theta)\|^2+2v^T\nabla_x (v^TS(x;\theta))\right] \\
    &=: L_{SSM}(\theta)
\end{align*}
$$


## Problem 2

> Briefly explain SDE.

### 1. SDE的核心定義

隨機微分方程 (Stochastic Differential Equation, SDE) 是一種用來描述系統如何隨時間演變的微分方程，但與普通微分方程 (ODE) 不同的是，它額外包含了一個隨機項。它主要用於模擬那些受到噪聲或隨機波動影響的動態系統

通用形式如下：

$$
dx_t = f(x_t, t)dt + G(x_t, t)dW_t, \quad x(0)=x_0
$$

微分方程包含兩項核心部分：

* **漂移項 (Drift Term)** $f(X_t, t)dt$ ：
    這是系統的**確定性**部分，描述了系統的平均趨勢或預期走向
  
* **擴散項 (Diffusion Term)** $G(X_t, t)dW_t$ ：
    這是系統的**隨機性**部分，用來模擬影響系統的波動性或噪聲

### 2. SDE 的組成

* **漂移項 (Drift Term)** $f(X_t, t)$ ：
    控制平均方向與速度
  
* **擴散項 (Diffusion Term)** $G(X_t, t)$ ：
    控制隨機擾動的強度。

* **隨機過程 (Stochastic Process)** $X_t$ ：
    系統狀態，隨時間變化

* **維納過程 (Wiener Process)** $W_t$ ：
    代表隨機噪音，也稱為**布朗運動 (Brownian motion)**，具有以下關鍵特性：
    * 路徑是連續的。
    * 增量（例如 $W_{t} - W_{s}$ ）服從常態分佈。
    * 在不重疊時間段內的增量是相互獨立的。

### 3. SDE的求解與分析

* **伊藤積分 (Itô Integral)**:
    SDE也可以寫成積分形式

    $$
    x_t = x_0 + \int^t_0 f(x_s, s)\,ds + \int^t_0 G(x_s, s)\,dW_s
    $$

* **求解方法**:
    * **解析解 (Analytical Solution)**: 對於一些結構相對簡單的 SDE，我們可以找到其精確的數學解，例如：
        * **纯漂移情況 (Pure drift)**: $dx_t = f(x_t, t)dt$
        * **純擴散情況 (Pure diffusion)**: $dx_t = G(x_t, t)dW_t$
    * **數值解 (Numerical Solution)**: 對於大多數複雜的 SDE，我們無法找到解析解。此時，我們會採用數值方法來模擬系統演化的可能路徑，如**歐拉-丸山法 (Euler-Maruyama method)**：
    
    $$
    X_{n+1}=X_{n}+f(X_{n}, t_n)\Delta t + G(X_{n}, t_n)\Delta W(t_n)
    $$


## Problem 3

> Unanswered Questions

1. 歐拉-丸山法是SDE最基礎的數值方法，它的收斂速度和精度如何？是否存在比它更精確、收斂更快的數值方法？
2. 既然SSM可以讓模型去學習高維度的數據，那有沒有可能用SSM製作一個動畫生成模型？（二維畫面+時間軸）
