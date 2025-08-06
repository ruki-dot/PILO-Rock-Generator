# 文件名: calculate_baseline_clean_output.py
# 最终版本：清理所有不必要的日志和进度条输出，保持结果整洁。

import os
import zipfile
import glob
import numpy as np
import porespy as ps
import openpnm as op
import matplotlib.pyplot as plt
from skimage.io import imread
import time
import shutil
import warnings
import logging
import sys
from contextlib import redirect_stdout

# ------------------- 1. 配置日志记录以屏蔽不必要的警告/错误 -------------------
# 只显示级别为CRITICAL或更高的日志，从而屏蔽掉ERROR和WARNING
logging.basicConfig(level=logging.CRITICAL)


# ------------------- 2. 功能函数定义 -------------------

def setup_matplotlib_for_chinese():
    """配置matplotlib以支持中文显示。"""
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        print("已配置matplotlib使用黑体。")
    except Exception:
        print("警告：未找到指定中文字体，将使用系统默认字体。")
        pass


def load_images_from_zip(zip_filename='sample_145_reconstructed_128_images.zip'):
    """从当前目录下的ZIP文件中加载图像。"""
    if not os.path.exists(zip_filename):
        raise FileNotFoundError(f"错误：在当前目录下找不到文件 '{zip_filename}'。")

    print(f"\n正在从 '{zip_filename}' 加载图像...")
    extract_dir = 'rock_images_local'
    if os.path.exists(extract_dir):
        shutil.rmtree(extract_dir)
    os.makedirs(extract_dir)

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    image_paths = sorted(glob.glob(os.path.join(extract_dir, '*.*')))
    if not image_paths:
        raise ValueError("在ZIP文件中没有找到任何图像文件。")

    print(f"找到 {len(image_paths)} 张图像...")
    volume_list = [imread(path, as_gray=True) for path in image_paths]
    volume_3d = np.stack(volume_list, axis=0)

    binary_volume = (volume_3d < np.mean(volume_3d)).astype(bool)

    print("三维数字岩心加载并二值化完成。")
    return binary_volume


def calculate_lbm_permeability_v2(im, axis, voxel_size=1e-6):
    """[已验证] 使用PoreSpy v2.4.2中存在的 tortuosity_fd 函数计算渗透率。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        # 暂时重定向输出来抑制LBM的进度条
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            result = ps.simulations.tortuosity_fd(im=im, axis=axis)

    tortuosity = result.tortuosity
    porosity = ps.metrics.porosity(im)

    if tortuosity == 0 or np.isinf(tortuosity) or np.isnan(tortuosity):
        K = 0.0
    else:
        K = (porosity * (voxel_size ** 2)) / tortuosity

    K_mD = K * 1e15
    return K_mD


def main():
    """主执行函数"""
    setup_matplotlib_for_chinese()

    try:
        rock_volume = load_images_from_zip()

        plt.figure(figsize=(6, 6))
        plt.imshow(rock_volume[len(rock_volume) // 2], cmap='gray')
        plt.title(f"二值化后的中心切片 (白色 = 孔隙)")
        plt.axis('off')
        plt.savefig("center_slice.png")
        print("中心切片图像已保存为 center_slice.png")

        print("\n--- 开始计算基线物理参数 ---")

        # 1. 计算孔隙度
        total_porosity = ps.metrics.porosity(rock_volume)
        print(f"1a. 总孔隙度 (Total Porosity) 计算完成: {total_porosity:.4f}")

        # 2. 计算有效孔隙度
        disconnected_voxels = ps.filters.find_disconnected_voxels(im=rock_volume)
        effective_im = rock_volume & ~disconnected_voxels
        effective_porosity = ps.metrics.porosity(effective_im)
        print(f"1b. 有效孔隙度 (Effective Porosity) 计算完成: {effective_porosity:.4f}")

        # 3. 计算平均配位数
        print("\n2. 正在提取孔隙网络以计算连通性...")
        # 暂时重定向输出来抑制snow2的进度条
        with open(os.devnull, 'w') as f, redirect_stdout(f):
            snow_output = ps.networks.snow2(effective_im)
        pn = op.io.network_from_porespy(snow_output.network)
        avg_coordination_number = np.mean(pn.num_neighbors(pn.pores()))
        print(f"   平均配位数 (Avg. Coordination #) 计算完成: {avg_coordination_number:.4f}")

        # 4. 计算各向异性渗透率
        print("\n3. 开始使用LBM在有效孔隙上计算渗透率...")

        start_time = time.time()

        print("   ...正在计算 X 方向渗透率 (Kx)...")
        Kx = calculate_lbm_permeability_v2(effective_im, axis=0, voxel_size=1e-6)
        print(f"   Kx 计算完成: {Kx:.4f} mD")

        print("   ...正在计算 Y 方向渗透率 (Ky)...")
        Ky = calculate_lbm_permeability_v2(effective_im, axis=1, voxel_size=1e-6)
        print(f"   Ky 计算完成: {Ky:.4f} mD")

        print("   ...正在计算 Z 方向渗透率 (Kz)...")
        Kz = calculate_lbm_permeability_v2(effective_im, axis=2, voxel_size=1e-6)
        print(f"   Kz 计算完成: {Kz:.4f} mD")

        end_time = time.time()
        print(f"渗透率计算总耗时: {end_time - start_time:.2f} 秒")

        print("--- 所有参数计算完毕 ---")

        properties = {
            "total_porosity": total_porosity,
            "effective_porosity": effective_porosity,
            "avg_coordination_number": avg_coordination_number,
            "permeability_Kx_mD": Kx,
            "permeability_Ky_mD": Ky,
            "permeability_Kz_mD": Kz,
        }

        with open("baseline_results_complete.txt", "w", encoding='utf-8') as f:
            f.write("数字岩心基线物理参数计算结果 (完整版)\n")
            f.write("=" * 50 + "\n")
            f.write(
                f"  总孔隙度 (Total Porosity)      : {properties['total_porosity']:.4f} (或 {properties['total_porosity'] * 100:.2f}%)\n")
            f.write(
                f"  有效孔隙度 (Effective Porosity) : {properties['effective_porosity']:.4f} (或 {properties['effective_porosity'] * 100:.2f}%)\n")
            f.write(f"  平均配位数 (Avg. Coord. #)     : {properties['avg_coordination_number']:.4f}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  X方向渗透率 (Kx)                : {properties['permeability_Kx_mD']:.4f} mD\n")
            f.write(f"  Y方向渗透率 (Ky)                : {properties['permeability_Ky_mD']:.4f} mD\n")
            f.write(f"  Z方向渗透率 (Kz)                : {properties['permeability_Kz_mD']:.4f} mD\n")
            f.write("=" * 50 + "\n")
        print("\n计算结果已保存到 baseline_results_complete.txt")

    except Exception as e:
        print(f"\n程序执行时发生错误: {e}")


if __name__ == "__main__":
    main()