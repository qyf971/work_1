import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# =========================
# 全局字体设置（新罗马）
# =========================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20
plt.rcParams['legend.fontsize'] = 20

# =========================
# 文件路径
# =========================
file1 = "dataset/Beijing_12/cleaned_data/Wanshouxigong.csv"
file2 = "dataset/Beijing_12/cleaned_data/Changping.csv"

# =========================
# 保存路径
# =========================
save_dir = "引言图绘制"
os.makedirs(save_dir, exist_ok=True)

# =========================
# 读取数据
# =========================
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# =========================
# 时间处理
# =========================
df1['date'] = pd.to_datetime(df1['date'])
df2['date'] = pd.to_datetime(df2['date'])

df1.set_index('date', inplace=True)
df2.set_index('date', inplace=True)

# =========================
# 选择某一个月
# =========================
year = 2013
month = 3

df1_month = df1[(df1.index.year == year) & (df1.index.month == month)].copy()
df2_month = df2[(df2.index.year == year) & (df2.index.month == month)].copy()

# =========================
# 缺失值处理
# =========================
df1_month.loc[:, 'PM2.5'] = df1_month['PM2.5'].interpolate()
df2_month.loc[:, 'PM2.5'] = df2_month['PM2.5'].interpolate()

# =========================
# 绘图
# =========================
fig, ax = plt.subplots(figsize=(12, 8))

ax.plot(df1_month.index, df1_month['PM2.5'], label='station A', linewidth=2) # 万寿西宫
ax.plot(df2_month.index, df2_month['PM2.5'], label='station C', linewidth=2) # 昌平

# y轴标签
ax.set_ylabel("PM2.5 (μg/m³)")

# =========================
# y轴刻度设置（每50递增）
# =========================
ax.yaxis.set_major_locator(MultipleLocator(50))

# 图例 & 网格
ax.legend()
ax.grid(alpha=0.3)

# x轴旋转
plt.xticks(rotation=45)

plt.tight_layout()

# =========================
# 保存图片
# =========================
save_path = os.path.join(save_dir, f"PM25_{year}_{month:02d}.png")
plt.savefig(save_path, dpi=300)

print(f"图像已保存到: {save_path}")

plt.show()