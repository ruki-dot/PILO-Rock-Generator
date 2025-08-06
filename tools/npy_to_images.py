import numpy as np
import os
from PIL import Image


def analyze_and_convert_npy(npy_path, output_dir):
    """
    加载一个包含三维或四维岩心数据的.npy文件，进行数据分析，
    并将其保存为一系列的2D PNG图像。

    - 自动处理3D (D, H, W) 或 4D (D, C, H, W) 数据。
    - 对解码后的数据进行统计分析，帮助确定合适的二值化阈值。
    - 使用数据的平均值作为动态阈值进行二值化。
    """
    print(f"--- 開始轉換: '{npy_path}' ---")

    # 1. 检查并创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: '{output_dir}'")
    else:
        # 清理旧文件，避免混淆
        print(f"警告: 输出目录 '{output_dir}' 已存在。将清空其中的内容。")
        for f in os.listdir(output_dir):
            file_path = os.path.join(output_dir, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
        print(f"已清理旧的输出目录。")

    # 2. 加载.npy文件
    try:
        rock_volume = np.load(npy_path)
        print(f"成功加载.npy文件，原始数据形状为: {rock_volume.shape}")
    except Exception as e:
        print(f"错误: 加载.npy文件失败: {e}")
        return

    # 3. 检查并处理4D数据
    if rock_volume.ndim == 4:
        print(f"检测到4D数据，形状为 (D, C, H, W)。")
        # 假设通道在第二个维度 (axis=1)
        if rock_volume.shape[1] > 1:
            print(f"  - 发现 {rock_volume.shape[1]} 个通道，将对通道进行平均以转换为灰度图...")
            rock_volume = np.mean(rock_volume, axis=1)
        else:  # 如果只有一个通道
            print("  - 发现1个通道，将移除单通道维度...")
            rock_volume = np.squeeze(rock_volume, axis=1)
        print(f"  - 数据已成功转换为3D，新形状为: {rock_volume.shape}")

    # 4. 再次检查，确保数据现在是3D的
    if rock_volume.ndim != 3:
        print(f"错误: 处理后数据维度仍不为3，无法继续。最终维度为 {rock_volume.ndim}。")
        return

    # 5. 对解码后的数据进行统计分析
    print("\n--- 数据分析 ---")
    min_val, max_val = np.min(rock_volume), np.max(rock_volume)
    mean_val, std_val = np.mean(rock_volume), np.std(rock_volume)
    print(f"  - 最小值 (Min)  : {min_val:.4f}")
    print(f"  - 最大值 (Max)  : {max_val:.4f}")
    print(f"  - 平均值 (Mean) : {mean_val:.4f}")
    print(f"  - 标准差 (Std)  : {std_val:.4f}")

    # 6. 使用平均值作为动态阈值进行二值化
    threshold = mean_val
    print(f"将使用动态阈值 (平均值) 进行二值化: {threshold:.4f}\n")

    # 7. 遍历三维数组并保存切片
    num_slices = rock_volume.shape[0]
    print(f"开始保存 {num_slices} 个切片...")

    for i in range(num_slices):
        slice_2d = rock_volume[i, :, :]

        # 将浮点数数组二值化为黑(0)白(255)图像
        binary_slice = (slice_2d > threshold).astype(np.uint8) * 255

        # 使用Pillow库从数组创建图像并保存
        img = Image.fromarray(binary_slice, 'L')  # 'L'表示8位灰度模式
        output_filename = os.path.join(output_dir, f"slice_{i:04d}.png")
        img.save(output_filename)

        # 打印进度
        if (i + 1) % 10 == 0 or (i + 1) == num_slices:
            print(f"  正在保存: {output_filename} ({i + 1}/{num_slices})")

    print(f"\n--- ✅ 转换完成！ ---")
    print(f"{num_slices} 张图像已保存至 '{output_dir}'")


# --- 主程序入口 ---
if __name__ == "__main__":
    # 使用os.path.join来构建路径，更具可移植性
    base_dir = './reconstructed_rocks'

    # !!! 用户可能需要修改这里的文件名 !!!
    # 根据reconstruct_rock.py的输出格式来确定
    # 例如：sample_145_reconstructed_128.npy
    input_npy_name = 'sample_145_reconstructed_128.npy'
    output_dir_name = os.path.splitext(input_npy_name)[0] + '_images'

    INPUT_NPY_FILE = os.path.join(base_dir, input_npy_name)
    OUTPUT_IMAGE_DIR = os.path.join(base_dir, output_dir_name)

    if not os.path.exists(INPUT_NPY_FILE):
        print(f"错误: 找不到输入文件 '{INPUT_NPY_FILE}'")
        print("请确保已成功运行 reconstruct_rock.py，并检查此脚本中的 `input_npy_name` 是否正确。")
    else:
        analyze_and_convert_npy(INPUT_NPY_FILE, OUTPUT_IMAGE_DIR)
