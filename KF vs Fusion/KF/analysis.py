import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# 读取数据
file_path = 'output_R_3.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# 初始化数据存储
R_values = []
KF_values = []
GPS_values = []

# 处理每行数据
for i in range(0, len(lines), 5):
    R = float(lines[i].strip())
    KF_mean = abs(float(lines[i + 1].split(',')[0]) + float(lines[i + 1].split(',')[1])) / 2
    KF_var = abs(float(lines[i + 2].split(',')[0]) + float(lines[i + 2].split(',')[1])) / 2
    GPS_mean = abs(float(lines[i + 3].split(',')[0]) + float(lines[i + 3].split(',')[1])) / 2
    GPS_var = abs(float(lines[i + 4].split(',')[0]) + float(lines[i + 4].split(',')[1])) / 2

    # 存储数据
    R_values.append(R)
    KF_values.append([KF_mean, KF_var])
    GPS_values.append([GPS_mean, GPS_var])

# 绘制箱线图
plt.figure(figsize=(12, 6))

# 遍历每个R值，画出KF和GPS的箱线图
for i, R in enumerate(R_values):
    positions = [i]

    # 使用不同颜色画箱线图和方差的上下拉线，并设置label
    bp_kf = plt.boxplot([KF_values[i]], positions=positions, showmeans=True,
                        boxprops=dict(color='dodgerblue', facecolor=(0.5, 0.5, 1, 0.5)),  # 设置底色，最后一个数字是透明度
                        medianprops=dict(color='dodgerblue'), whiskerprops=dict(color='dodgerblue'), widths=0.6, labels=['KF'],
                        patch_artist=True)  # 使用 patch_artist=True

    bp_gps = plt.boxplot([GPS_values[i]], positions=positions, showmeans=True,
                         boxprops=dict(color='darkorange', facecolor=(1, 0.8, 0.6, 0.5)),  # 设置底色，最后一个数字是透明度
                         medianprops=dict(color='darkorange'), whiskerprops=dict(color='darkorange'), widths=0.6, labels=['GPS'],
                         patch_artist=True)  # 使用 patch_artist=True

    # 设置箱体颜色
    # for box in (bp_kf['boxes'] + bp_gps['boxes']):
    #     box.set_facecolor('none')

# 设置 x 轴刻度和标签
plt.xticks(range(len(R_values)), [f'R={R}' for R in R_values])
# 添加图例
kf_patch = mpatches.Patch(color=(0.5, 0.5, 1, 0.5), label='KF (Kalman Filter)')  # 设置颜色
gps_patch = mpatches.Patch(color=(1, 0.8, 0.6, 0.5), label='GPS')  # 设置颜色

# 设置图例位置
plt.legend(handles=[kf_patch, gps_patch], loc='upper right')

plt.title("KF and GPS Differences for Different R Values", y=1.05)
plt.xlabel('R Values')
plt.ylabel('Values')

plt.show()