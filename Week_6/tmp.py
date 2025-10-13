import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# --- 手動實現 sklearn 的功能 ---

def manual_train_test_split(X, y, test_size=0.2, random_state=None):
    """
    手動實現數據集分割功能。
    """
    if random_state:
        np.random.seed(random_state)
    
    # 產生一個隨機排列的索引序列
    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)
    
    # 計算分割點
    split_idx = int(num_samples * (1 - test_size))
    
    # 分割索引
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # 使用索引來分割數據
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    return X_train, X_test, y_train, y_test

class ManualStandardScaler:
    """
    手動實現數據標準化功能。
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, data):
        """計算並儲存數據的均值和標準差"""
        self.mean_ = np.mean(data, axis=0)
        self.std_ = np.std(data, axis=0)
        # 為了避免除以零，將標準差為零的位置替換成一個極小值
        self.std_[self.std_ == 0] = 1e-8
        return self

    def transform(self, data):
        """使用已計算的均值和標準差來標準化數據"""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (data - self.mean_) / self.std_

    def inverse_transform(self, data):
        """將標準化後的數據還原成原始尺度"""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Scaler has not been fitted yet.")
        return (data * self.std_) + self.mean_


# --- 1. 生成與準備數據 ---
def generate_temperature_map(num_samples=1500):
    np.random.seed(42)
    X = np.random.rand(num_samples, 2) * 10
    hot_spot, cold_spot = np.array([5, 5]), np.array([1, 8])
    dist_hot = np.linalg.norm(X - hot_spot, axis=1)
    dist_cold = np.linalg.norm(X - cold_spot, axis=1)
    base_temp = 15.0
    temp_from_hot = 20 * np.exp(-dist_hot**2 / 5)
    temp_from_cold = -15 * np.exp(-dist_cold**2 / 8)
    gradient = 0.5 * X[:, 0]
    noise = np.random.randn(num_samples) * 0.5
    y = base_temp + temp_from_hot + temp_from_cold + gradient + noise
    return X, y.reshape(-1, 1)

X, y = generate_temperature_map()
# 使用我們手動實現的函數
X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2, random_state=42)

# 使用我們手動實現的類別
scaler_X = ManualStandardScaler().fit(X_train)
scaler_y = ManualStandardScaler().fit(y_train)

X_train_scaled = scaler_X.transform(X_train)
X_train_tensor = torch.from_numpy(X_train_scaled).float()
y_train_tensor = torch.from_numpy(scaler_y.transform(y_train)).float()

# --- 2. 定義回歸神經網路模型 (與之前相同) ---
class TemperatureRegressor(nn.Module):
    def __init__(self):
        super(TemperatureRegressor, self).__init__()
        self.layer1 = nn.Linear(2, 32)
        self.layer2 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

model = TemperatureRegressor()

# --- 3. 修改訓練迴圈以儲存快照 ---
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

epochs = 300
snapshot_interval = 5 # 每隔 5 個 epoch 儲存一次快照
snapshots = []
epoch_numbers = []

# 創建一個固定的網格用於預測，以確保動畫的連續性
grid_x = np.linspace(0, 10, 100)
grid_y = np.linspace(0, 10, 100)
xx, yy = np.meshgrid(grid_x, grid_y)
X_grid_orig = np.c_[xx.ravel(), yy.ravel()]
X_grid_scaled = scaler_X.transform(X_grid_orig)
grid_tensor = torch.from_numpy(X_grid_scaled).float()

print("開始訓練並捕捉快照...")
for epoch in tqdm(range(epochs)):
    model.train()
    y_pred = model(X_train_tensor)
    loss = criterion(y_pred, y_train_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 儲存快照
    if (epoch + 1) % snapshot_interval == 0 or epoch == 0:
        model.eval()
        with torch.no_grad():
            pred_scaled = model(grid_tensor)
            # 逆轉換回真實溫度並儲存
            zz_pred = scaler_y.inverse_transform(pred_scaled.numpy()).reshape(xx.shape)
            snapshots.append(zz_pred)
            epoch_numbers.append(epoch + 1)

print(f"訓練完成！共捕捉了 {len(snapshots)} 個快照。")

# --- 4. 創建並儲存動畫 ---
print("正在生成 GIF 動畫...")

fig, ax = plt.subplots(figsize=(8, 7))

# 繪製第一幀作為基礎
# levels 參數確保所有幀的顏色條範圍一致
levels = np.linspace(np.min(snapshots), np.max(snapshots), 50)
contour = ax.contourf(xx, yy, snapshots[0], levels=levels, cmap='inferno')
ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
fig.colorbar(contour, ax=ax, label='Predicted Temperature (°C)')

# 動畫更新函數：每一幀都會呼叫這個函數
def update(frame):
    ax.clear() # 清除上一幀的內容
    # 繪製新一幀的內容
    contour = ax.contourf(xx, yy, snapshots[frame], levels=levels, cmap='inferno')
    ax.set_title(f"Temperature Map Prediction\nEpoch: {epoch_numbers[frame]}", fontsize=16)
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    # 為了保持顏色條穩定，我們不在这里重新創建它，但這也意味著它不會被清除。
    # 對於 `contourf`，這通常是可接受的。

# 創建動畫物件
# frames 是快照的數量，interval 是每幀之間的延遲（毫秒）
ani = FuncAnimation(fig, update, frames=len(snapshots), interval=100)

# 儲存為 GIF
# 需要 pillow writer。fps (Frames Per Second) 控制動畫播放速度。
ani.save('training_animation.gif', writer='pillow', fps=15)

plt.close() # 關閉圖形介面，因為我們已經儲存了文件
print("動畫已成功儲存為 'training_animation.gif'")

