import os
import numpy as np
import matplotlib.pyplot as plt

# =========================
# 路径设置
# =========================
pred_root = "RLMC_final_预测结果/proposed"
true_root = "RLMC_final_数据集/proposed"

save_dir = "绘制预测结果/预测结果图片"
os.makedirs(save_dir, exist_ok=True)

horizons = [1, 3, 6]
plot_range = 150

# =========================
# 站点名称
# =========================
stations = [
    "Aoizhongxin", "Changping", "Dingling", "Dongsi",
    "Guanyuan", "Gucheng", "Huairou", "Nongzhanguan",
    "Shunyi", "Tiantan", "Wanliu", "Wanshouxigong"
]

# stations = [
#     "奥体中心",   # Aoizhongxin
#     "昌平",       # Changping
#     "定陵",       # Dingling
#     "东四",       # Dongsi
#     "观园",       # Guanyuan
#     "古城",       # Gucheng
#     "怀柔",       # Huairou
#     "农展馆",     # Nongzhanguan
#     "顺义",       # Shunyi
#     "天坛",       # Tiantan
#     "万柳",       # Wanliu
#     "万寿西宫"    # Wanshouxigong
# ]

# =========================
# 字体
# =========================
plt.rcParams['font.family'] = 'Times New Roman'

# =========================
# 画图函数
# =========================
def plot_one_horizon(h):
    pred_path = os.path.join(pred_root, str(h), "final_pred.npy")
    true_path = os.path.join(true_root, str(h), "test_y_inverse.npy")

    pred = np.load(pred_path)
    true = np.load(true_path)

    # 取最后一步
    pred = pred[:, :, -1]
    true = true[:, :, -1]

    # 截取
    pred = pred[:plot_range]
    true = true[:plot_range]

    # 统一y轴
    y_min = min(true.min(), pred.min())
    y_max = max(true.max(), pred.max())

    fig, axes = plt.subplots(3, 4, figsize=(15, 10))
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]

        # 曲线
        ax.plot(true[:, i], color='#0000ff', linewidth=1.5, label='True')
        ax.plot(pred[:, i], color='black', linestyle='--', linewidth=1.5, label='DMGENet')

        # 标题
        ax.set_title(f"({chr(97+i)}) {stations[i]}", fontsize=16)

        # 坐标轴标签
        if i % 4 == 0:
            ax.set_ylabel("PM$_{2.5}$ (μg/m³)", fontsize=16)
        if i >= 8:
            ax.set_xlabel("Times (h)", fontsize=16)

        # y轴统一
        ax.set_ylim(y_min, y_max)

        # 横坐标刻度（每25）
        ax.set_xticks(np.arange(0, plot_range + 1, 25))

        # 网格（虚线）
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)

        # 完整边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1.0)

        # 刻度朝内
        ax.tick_params(direction='in', labelsize=10)

        # 图例（带边框）
        ax.legend(frameon=True, fontsize=10, loc='upper right')

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"plot_horizon_{h}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved: {save_path}")


# =========================
# 主程序
# =========================
if __name__ == "__main__":
    for h in horizons:
        plot_one_horizon(h)