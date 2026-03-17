import os
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from resnet50_train_NMM import ResNet50, standardize
import matplotlib.pyplot as plt  # 导入matplotlib库
import json  # 导入json模块
from sklearn.manifold import MDS
from scipy.optimize import minimize
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import multiprocessing as mp
from functools import partial
from scipy.spatial.distance import pdist

# 添加读取statistics.json文件的函数
def load_statistics(file_path):
    with open(file_path, 'r') as f:
        stats = json.load(f)
    return stats['total_mean'], stats['total_std_dev'], stats['total_mean_labels'], stats['total_std_dev_labels']

# 读取标准化参数
mean, std, mean_labels, std_labels = load_statistics(file_path='statistics.json')
# print(f'the mean and std of the scattering and labels are: {mean},{std},{mean_labels},{std_labels}')
std_labels = torch.tensor(std_labels)
mean_labels = torch.tensor(mean_labels)

def load_data_for_inference(batch_file):
    data = np.load(batch_file)
    inputs = torch.tensor(data['scattering'], dtype=torch.float32).unsqueeze(1)
    labels = torch.tensor(data['labels'], dtype=torch.float32)
    return inputs, labels

# 添加读取每个样本的statistics.json文件的函数
def load_sample_statistics(file_path):
    with open(file_path, 'r') as f:
        sample_stats = json.load(f)
    return sample_stats['mean_patterns'], sample_stats['std_dev_patterns']

# 修改load_data_for_experimet函数以加载每个样本的统计信息
def load_data_for_experimet(batch_file):
    data = np.load(batch_file)
    inputs = torch.tensor(data['scattering'], dtype=torch.float32).unsqueeze(1)
    return inputs

# 添加逆标准化函数
def inverse_standardize_labels(labels, mean=mean_labels, std=std_labels):
    return labels * std + mean

def inference(model, data_loader, device):
    print("开始进行推理")
    model.eval()  # 设置模型为评估模式
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
        print("推理完成")

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    # 逆标准化预测结果和真实标签
    predictions = inverse_standardize_labels(torch.tensor(predictions))

    # 计算差值
    differences = predictions - true_labels

    # 打印预测结果和真实标签
    for pred, true, diff in zip(predictions, true_labels, differences):
        print(f'Predicted: {pred}, True: {true}, Difference: {diff}')

    return predictions, true_labels, differences  # 返回差值

# 添加一个新的推理函数针对实验数据进行推理
def inference_experiment(model, data_loader, device, mean,std):
    print("开始进行实验数据推理")
    model.eval()  # 设置模型为评估模式
    n_patterns = 45
    predictions = {}  # 修改为字典

    with torch.no_grad():
        for index, inputs in enumerate(data_loader):
            inputs = inputs[0].to(device).squeeze()  # 解包inputs元组
            print(f'the shape of the {index+1}th batch is {inputs.shape}')
            # 按照行数拆分成35次推理
            for i in range(inputs.size(0)):
            # for i in range(3):
                single_input = inputs[i].unsqueeze(0).unsqueeze(0)  # 增加两个维度
                # print(f'the shape of the {i}th pattern in the {index+1}th batch is {single_input.shape}')
                single_input = (single_input - mean) / std  # 标准化
                output = model(single_input)
                if index not in predictions:
                    predictions[index] = {}
                predictions[index][i] = output.cpu().numpy()  # 修改赋值方式
                predictions[index][i] = inverse_standardize_labels(torch.tensor(predictions[index][i])).numpy()
                #print(f'the result of {i}th pattern in the {index+1}th batch is {predictions[index][i]}')
            print(f'推理第{index+1}批数据')
        print("实验数据推理完成")

    # # 逆标准化预测结果
    # for index in predictions:
    #     for i in predictions[index]:
    #         predictions[index][i] = inverse_standardize_labels(torch.tensor(predictions[index][i])).numpy()

    # 计算均值和标准差
    mean_predictions = np.zeros((n_patterns, 23))
    std_predictions = np.zeros((n_patterns, 23))

    for i in range(n_patterns):
        pattern_predictions = np.array([predictions[index][i] for index in predictions if i in predictions[index]])
        mean_predictions[i] = np.mean(pattern_predictions, axis=0)
        std_predictions[i] = np.std(pattern_predictions, axis=0)

    np.savez(f'./NMM_predictions.npz', mean=mean_predictions, std=std_predictions)

    # print(f'the result angle of the all patterns is {mean_predictions[:, 4]}')
    # # 绘制预测结果
    # plt.figure(figsize=(15, 10))
    # labels = ["rCH", "rCI", "rHI", "rHH", "cos_angle"]   #         label = np.array([r_CH, r_CI, r_HI_avg, r_HH_avg, cos_angle])
    # for j in range(23):
    #     plt.errorbar(range(n_patterns), mean_predictions[:, j], yerr=std_predictions[:, j], label=labels[j], fmt='-o')
    # plt.xlabel('Pattern Index')
    # plt.ylabel('Predicted Value')
    # plt.title('Predicted Values for Each Pattern')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return predictions

def plot_predictions(selected_labels=None, prediction_file='NMM_predictions.npz', n_patterns=45):
    """
    从保存的预测结果文件中读取数据，并绘制选定的labels的预测值
    
    参数:
        prediction_file: 保存预测结果的文件路径
        selected_labels: 要绘制的labels的索引列表，如果为None则默认绘制前5个
        n_patterns: pattern的数量
    """
    # 加载预测结果
    data = np.load(prediction_file)
    mean_predictions = data['mean']
    std_predictions = data['std']
    angle2 = np.arcsin(mean_predictions[:, 21])
    angle1 = np.arccos(mean_predictions[:, 22])
    plt.figure()
    plt.plot(mean_predictions[:, 21], 'o')
    plt.xlabel('angle1')
    plt.ylabel('angle2')
    plt.show()

    plt.figure()
    plt.plot(mean_predictions[:, 22], 'o')
    plt.xlabel('angle1')
    plt.ylabel('angle2')
    plt.show()
    
    # 如果没有指定要绘制的labels，则默认绘制前5个
    if selected_labels is None:
        selected_labels = list(range(5))
    
    # 定义所有labels的名称
    all_labels = ["rCH", "rCI", "rHI", "rC1C5", "cos_angle"] + [f"label_{i}" for i in range(5, 23)]
    
    # 创建图形
    plt.figure(figsize=(15, 10))
    
    # 为每个选定的label绘制误差条
    for j in selected_labels:
        plt.errorbar(range(n_patterns), 
                    mean_predictions[:, j], 
                    yerr=std_predictions[:, j], 
                    label=all_labels[j], 
                    fmt='-o',
                    capsize=5)
    
    plt.xlabel('Pattern Index')
    plt.ylabel('Predicted Value')
    plt.title('Predicted Values for Selected Labels')
    plt.legend()
    plt.grid(True)
    plt.show()

def build_distance_matrix(distances, n_atoms=7):
    """从距离列表构建距离矩阵"""
    distance_matrix = np.zeros((n_atoms, n_atoms))
    idx = 0
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            distance_matrix[i, j] = distances[idx]
            distance_matrix[j, i] = distances[idx]
            idx += 1
    return distance_matrix

def mds_error(distances, n_atoms=7):
    """计算MDS重构误差"""
    distance_matrix = build_distance_matrix(distances, n_atoms)
    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(distance_matrix)
    
    # 计算重构误差
    reconstructed_distances = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            reconstructed_distances[i,j] = reconstructed_distances[j,i] = np.linalg.norm(coords[i] - coords[j])
    
    return np.sum((distance_matrix - reconstructed_distances)**2)

def optimize_single_trial(args):
    """单次优化尝试"""
    mean_distances, std_distances, bounds, n_atoms = args
    x0 = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds])
    
    def objective_function(x):
        """目标函数：MDS重构误差 + 距离偏离惩罚"""
        if std_distances is not None:
            # 添加距离偏离惩罚项
            penalty = np.sum((x - mean_distances)**2 / (std_distances**2 + 1e-6))
            return mds_error(x, n_atoms) + penalty
        else:
            return mds_error(x, n_atoms)
    
    result = minimize(objective_function, x0, bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

def reconstruct_coords_from_labels(mean_distances, std_distances=None, n_trials=10):
    """
    使用多维尺度分析(MDS)从距离数据重建NMM分子坐标，考虑标准差信息进行优化
    参数:
        mean_distances: 21个原子对距离的均值
        std_distances: 21个原子对距离的标准差（可选）
        n_trials: 优化尝试次数，建议设置为10-20次
    返回:
        best_coords: 最优的分子坐标 (7,3)
        best_error: 最优解的重构误差
    """    
    n_atoms = 7
    
    # 设置优化边界
    bounds = []
    for i in range(21):
        if std_distances is not None:
            # 允许距离在均值±3倍标准差范围内变化
            lower = max(0.1, mean_distances[i] - 3*std_distances[i])
            upper = mean_distances[i] + 3*std_distances[i]
        else:
            # 如果没有标准差信息，允许在±10%范围内变化
            lower = max(0.1, mean_distances[i] * 0.9)
            upper = mean_distances[i] * 1.1
        bounds.append((lower, upper))
    
    # 使用多进程进行优化
    n_processes = min(mp.cpu_count(), n_trials)  # 使用不超过CPU核心数的进程
    with mp.Pool(processes=n_processes) as pool:
        results = pool.map(optimize_single_trial, 
                         [(mean_distances, std_distances, bounds, n_atoms) for _ in range(n_trials)])
    
    # 找出最优结果
    best_error = float('inf')
    best_distances = None
    
    for distances, error in results:
        if error < best_error:
            best_error = error
            best_distances = distances
    
    # 使用最优距离重建坐标
    if best_distances is not None:
        distance_matrix = build_distance_matrix(best_distances, n_atoms)
        mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
        best_coords = mds.fit_transform(distance_matrix)
    else:
        best_coords = None
    
    return best_coords, best_error

def align_structures_with_ring(reference, target):
    """
    使用六元环上的原子（C1, C2, C4, C6）和O7进行对齐
    参数:
        reference: 参考结构 (7,3)
        target: 目标结构 (7,3)
    返回:
        aligned: 对齐后的目标结构
    """
    # 选择用于对齐的原子索引（C1, C2, C4, C6, O7）
    align_indices = [0, 1, 3, 5, 6]  # 对应C1, C2, C4, C6, O7
    
    # 提取用于对齐的原子坐标
    ref_align = reference[align_indices]
    target_align = target[align_indices]
    
    # 计算质心
    ref_centroid = np.mean(ref_align, axis=0)
    target_centroid = np.mean(target_align, axis=0)
    
    # 中心化
    ref_centered = ref_align - ref_centroid
    target_centered = target_align - target_centroid
    
    # 计算旋转矩阵
    H = target_centered.T @ ref_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # 处理反射情况
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = Vt.T @ U.T
    
    # 应用旋转和平移到整个结构
    target_centered_all = target - target_centroid
    aligned = (target_centered_all @ R) + ref_centroid
    
    return aligned

def reconstruct_all_timepoints(prediction_file='NMM_predictions.npz', output_file='NMM_reconstructed_coords.npz'):
    """
    从预测结果文件中读取所有时刻的数据，进行坐标反演，并保存结果
    参数:
        prediction_file: 预测结果文件路径
        output_file: 输出文件路径
    """
    # 加载预测结果
    data = np.load(prediction_file)
    mean_predictions = data['mean']  # shape: (45, 23)
    std_predictions = data['std']    # shape: (45, 23)
    
    n_timepoints = mean_predictions.shape[0]
    all_coords = np.zeros((n_timepoints, 7, 3))  # 存储所有时刻的坐标
    all_errors = np.zeros(n_timepoints)          # 存储所有时刻的重构误差
    
    # 对每个时刻进行坐标反演
    for t in range(n_timepoints):
        print(f"处理第 {t+1} 个时刻...")
        mean_distances = mean_predictions[t, :21]  # 21个距离均值
        std_distances = std_predictions[t, :21]    # 21个距离标准差
        
        # 重建坐标
        coords, error = reconstruct_coords_from_labels(mean_distances, std_distances)
        
        if coords is not None:
            all_coords[t] = coords
            all_errors[t] = error
        else:
            print(f"警告：第 {t+1} 个时刻的坐标重建失败")
    
    # 对齐所有结构到第一个时间点的结构，使用环上的原子
    print("开始对齐所有分子结构...")
    reference = all_coords[0]
    for t in range(1, n_timepoints):
        all_coords[t] = align_structures_with_ring(reference, all_coords[t])
    
    # 保存结果
    np.savez(output_file, 
             coords=all_coords,      # 所有时刻的分子坐标
             errors=all_errors,      # 所有时刻的重构误差
             mean_distances=mean_predictions[:, :21],  # 所有时刻的距离均值
             std_distances=std_predictions[:, :21])    # 所有时刻的距离标准差
    
    print(f"坐标重建完成，结果已保存到 {output_file}")
    
    # 绘制重构误差随时间的变化
    plt.figure(figsize=(10, 5))
    plt.plot(range(n_timepoints), all_errors, 'o-')
    plt.xlabel('时间点')
    plt.ylabel('重构误差')
    plt.title('坐标重构误差随时间的变化')
    plt.grid(True)
    plt.savefig('reconstruction_errors.png')
    plt.close()
    
    return all_coords, all_errors

def create_molecule_animation(coords_file='NMM_reconstructed_coords.npz', 
                            output_file='molecule_animation.gif',
                            time_intervals=None,
                            fps=10):
    """
    创建分子坐标的动画
    参数:
        coords_file: 坐标文件路径
        output_file: 输出动画文件路径
        time_intervals: 时间间隔列表
        fps: 动画帧率
    """
    
    # 加载坐标数据
    data = np.load(coords_file)
    coords = data['coords']  # shape: (45, 7, 3)
    
    # 如果没有指定时间间隔，使用默认值
    if time_intervals is None:
        time_intervals = [5] * coords.shape[0]  # 默认每帧间隔0.5秒
    
    # 创建图形
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    x_min, x_max = np.min(coords[:, :, 0]), np.max(coords[:, :, 0])
    y_min, y_max = np.min(coords[:, :, 1]), np.max(coords[:, :, 1])
    z_min, z_max = np.min(coords[:, :, 2]), np.max(coords[:, :, 2])
    
    # 添加一些余量
    margin = 0.5
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    # 设置标签
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    
    # 定义原子标签
    atom_labels = ['C1', 'C2', 'N3', 'C4', 'C5', 'C6', 'O7']
    
    # 定义原子颜色和大小
    atom_colors = ['gray'] * 7  # 默认所有原子为灰色
    atom_sizes = [200] * 7      # 默认所有原子大小相同
    
    # 初始化散点图
    scatter = ax.scatter([0]*7, [0]*7, [0]*7, c=atom_colors, s=atom_sizes)
    
    # 初始化原子标签文本
    label_texts = []
    for i in range(7):
        text = ax.text(0, 0, 0, atom_labels[i], fontsize=12)
        label_texts.append(text)
    
    # 初始化线
    lines = []
    for i in range(7):
        for j in range(i+1, 7):
            line, = ax.plot([], [], [], 'k-', alpha=0.3)
            lines.append(line)
    
    # 初始化时间文本
    time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)
    
    def init():
        scatter._offsets3d = ([0]*7, [0]*7, [0]*7)
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        for text in label_texts:
            text.set_position((0, 0, 0))
        time_text.set_text('')
        return [scatter] + lines + label_texts + [time_text]
    
    def update(frame):
        # 更新散点图
        scatter._offsets3d = (coords[frame, :, 0], 
                            coords[frame, :, 1], 
                            coords[frame, :, 2])
        
        # 更新原子标签位置
        for i, text in enumerate(label_texts):
            text.set_position((coords[frame, i, 0], 
                             coords[frame, i, 1], 
                             coords[frame, i, 2]))
        
        # 更新线
        line_idx = 0
        for i in range(7):
            for j in range(i+1, 7):
                lines[line_idx].set_data([coords[frame, i, 0], coords[frame, j, 0]],
                                       [coords[frame, i, 1], coords[frame, j, 1]])
                lines[line_idx].set_3d_properties([coords[frame, i, 2], coords[frame, j, 2]])
                line_idx += 1
        
        # 更新时间文本
        time_text.set_text(f'Time: {sum(time_intervals[:frame+1]):.1f} ps')
        
        return [scatter] + lines + label_texts + [time_text]
    
    # 创建动画
    anim = FuncAnimation(fig, update, frames=coords.shape[0],
                        init_func=init, blit=True,
                        interval=10000/fps)  # interval in milliseconds
    
    # 保存动画
    anim.save(output_file, writer='pillow', fps=fps)
    print(f"动画已保存到 {output_file}")
    
    plt.close()

def realign_saved_coordinates(input_file='NMM_reconstructed_coords.npz', output_file='NMM_reconstructed_coords_aligned.npz'):
    """
    重新处理已保存的坐标数据，实现更严格的对齐
    参数:
        input_file: 输入的坐标文件
        output_file: 输出的坐标文件
    """
    # 加载坐标数据
    data = np.load(input_file)
    coords = data['coords']  # shape: (45, 7, 3)
    errors = data['errors']
    
    # 选择用于对齐的原子索引（C1, C2, C4, C6）
    ring_indices = [0, 1, 3, 5]  # 对应C1, C2, C4, C6
    
    # 使用第一个结构作为参考
    reference = coords[0]
    aligned_coords = np.zeros_like(coords)
    aligned_coords[0] = reference
    
    # 计算参考环的法向量
    ref_ring = reference[ring_indices]
    ref_centroid = np.mean(ref_ring, axis=0)
    ref_vectors = ref_ring - ref_centroid
    ref_normal = np.cross(ref_vectors[1] - ref_vectors[0], 
                         ref_vectors[2] - ref_vectors[0])
    ref_normal = ref_normal / np.linalg.norm(ref_normal)
    
    print("开始重新对齐所有分子结构...")
    for t in range(1, coords.shape[0]):
        current = coords[t]
        
        # 提取当前环的原子
        current_ring = current[ring_indices]
        current_centroid = np.mean(current_ring, axis=0)
        
        # 计算当前环的法向量
        current_vectors = current_ring - current_centroid
        current_normal = np.cross(current_vectors[1] - current_vectors[0],
                                current_vectors[2] - current_vectors[0])
        current_normal = current_normal / np.linalg.norm(current_normal)
        
        # 计算旋转矩阵
        v = np.cross(current_normal, ref_normal)
        c = np.dot(current_normal, ref_normal)
        s = np.linalg.norm(v)
        
        if s < 1e-6:
            # 如果法向量几乎平行，不需要旋转
            R = np.eye(3)
        else:
            # 使用Rodrigues旋转公式
            v = v / s
            vx = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 - c) / (s * s)
        
        # 应用旋转和平移
        centered = current - current_centroid
        rotated = centered @ R.T
        aligned = rotated + ref_centroid
        
        # 检查环的朝向是否一致
        aligned_ring = aligned[ring_indices]
        aligned_vectors = aligned_ring - ref_centroid
        aligned_normal = np.cross(aligned_vectors[1] - aligned_vectors[0],
                                aligned_vectors[2] - aligned_vectors[0])
        aligned_normal = aligned_normal / np.linalg.norm(aligned_normal)
        
        # 如果法向量方向相反，需要翻转
        if np.dot(aligned_normal, ref_normal) < 0:
            aligned = 2 * ref_centroid - aligned
        
        aligned_coords[t] = aligned
    
    # 保存重新对齐后的坐标
    np.savez(output_file,
             coords=aligned_coords,
             errors=errors,
             mean_distances=data['mean_distances'],
             std_distances=data['std_distances'])
    
    print(f"重新对齐完成，结果已保存到 {output_file}")
    return aligned_coords, errors

# 修改main函数以适应新的load_data_for_experimet函数
if __name__ == '__main__':
    # 加载实验数据
    batch_file = 'pattern_experiments_NMM.npz'  # 新的实验数据文件 压缩结构
    inputs = load_data_for_experimet(batch_file)
    # 创建一个包含inputs和sample_stats的自定义Dataset
    dataset = TensorDataset(inputs)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=5)

    model = ResNet50(input_channels=1, output_dim=23)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if os.path.exists('./resnet50_model_NMM.pth'):
        # model.load_state_dict(torch.load('./resnet50_model_NMM.pth', map_location=device))
        # model.load_state_dict(torch.load('./params/resnet50_model_label_nostd.pth', map_location=device))
        print("已加载已保存模型参数")
    else:
        print("未找到模型参数文件，无法进行推理")

    # predictions = inference_experiment(model, data_loader, device, mean,std)  # 使用新的推理函数

    # plot_predictions(selected_labels=[3,8,12,15,18,19,21,22], prediction_file='NMM_predictions.npz', n_patterns=45)

    # 重建所有时刻的坐标
    # all_coords, all_errors = reconstruct_all_timepoints(prediction_file='NMM_predictions.npz')
    
    # 重新对齐已保存的坐标
    aligned_coords, errors = realign_saved_coordinates()
    
    # 使用重新对齐后的坐标创建动画
    time_intervals = [2]*30 + [4]*15
    create_molecule_animation(coords_file='NMM_reconstructed_coords_aligned.npz',
                            time_intervals=time_intervals)

    # # 绘制直方图
    # plt.figure(figsize=(15, 5))
    # for i in range(3):
    #     plt.subplot(1, 3, i+1)
    #     plt.hist(predictions[:, i], bins=20, color='blue', alpha=0.7)
    #     plt.title(f'Histogram of {"rCS1" if i == 0 else "rCS2" if i == 1 else "rSS"} Differences')
    #     plt.xlabel('Difference')
    #     plt.ylabel('Frequency')
    # plt.tight_layout()
    # plt.show()
