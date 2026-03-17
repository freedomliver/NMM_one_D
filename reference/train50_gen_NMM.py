import numpy as np
import scipy.io as sio
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time
import os
import json
import gc  # 导入垃圾回收模块


def generate_nmm_structure_variation(base_coords, n_samples=1000, displacement_std=0.005, c5_displacement_std=0.001):
    """
    从基础NMM结构生成多个扰动结构，用于CNN训练。
    参数:  base_coords: 7x3 NMM骨架坐标 (不包含H)  n_samples: 生成样本数  displacement_std: 坐标扰动强度 (单位 Å)  
    返回:  coords_list: 所有生成的分子坐标 shape=(n_samples, 7, 3)  labels_list: 所有原子对距离标签 shape=(n_samples, 23)
    """
    coords_list = []
    labels_list = []

    for _ in range(n_samples):
        displacement = np.random.normal(0, displacement_std, base_coords.shape)          # 添加高斯随机扰动（键长、面外振动）
        displacement[4] = np.random.normal(0, c5_displacement_std, 3)
        coords = base_coords + displacement

        # 计算21个原子对距离
        distances = pdist(coords)
        C2 = coords[1] # 计算out-of-plane角度
        N3 = coords[2]        # 获取C2, N3, C4的坐标
        C4 = coords[3]
        C5 = coords[4]
        
        v1 = C2 - N3          # 计算平面法向量
        v2 = C4 - N3
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        v3 = C5 - N3          # 计算C5-N3向量
        v3 = v3 / np.linalg.norm(v3)
        
        angle = np.arcsin((np.dot(v3, normal)))
        sin_angle = np.sin(angle)        # 计算夹角（弧度）
        cos_angle = np.cos(angle)
        
        # 组合标签：21个距离 + sin(angle) + cos(angle)
        label = np.concatenate([distances, [sin_angle, cos_angle]])
        
        coords_list.append(coords)
        labels_list.append(label)

    np.savez('/Users/jinc_air/Documents/ML_UED/NMM/nmm_structure_eq_10000.npz', labels=labels_list, coords=coords_list)
    print(f'Generated {len(labels_list)} NMM structure labels.')

    return np.array(coords_list), np.array(labels_list)

def generate_NMM_structure(base_coords, n_samples=1000, displacement_std=0.001, angle_range=(-50, 50)):
    """
    生成NMM分子的结构变化
    参数:
        base_coords: 基础坐标 (7,3)
        n_samples: 生成样本数
        displacement_std: 坐标扰动强度 (单位 Å)，设为0表示不考虑结构扰动
        angle_range: out-of-plane角度范围 (度)，设为(0,0)表示不考虑角度变化
    返回:
        coords_list: 所有生成的分子坐标 shape=(n_samples, 7, 3)
        labels_list: 所有标签 shape=(n_samples, 23)
    """
    coords_list = []
    labels_list = []
    angle_range_rad = np.radians(angle_range)    # 将角度范围转换为弧度
    C2_base = base_coords[1]     # 获取基础结构中的关键原子坐标
    N3_base = base_coords[2]
    C4_base = base_coords[3]
    C5_base = base_coords[4]
    # 计算基础平面法向量
    v1_base = C2_base - N3_base
    v2_base = C4_base - N3_base
    normal_base = np.cross(v1_base, v2_base)
    normal_base = normal_base / np.linalg.norm(normal_base)
    # 计算基础C5-N3向量
    v3_base = C5_base - N3_base
    v3_base = v3_base / np.linalg.norm(v3_base)
    # 计算基础out-of-plane角度
    base_angle = np.arcsin(np.abs(np.dot(v3_base, normal_base)))
    
    for i in range(n_samples):
        coords = base_coords.copy()
        displacement = np.random.normal(0, displacement_std, base_coords.shape)
        coords += displacement        # 添加结构扰动
        
        if angle_range[0] != angle_range[1]:        # 添加out-of-plane振动
            target_angle = np.random.uniform(*angle_range_rad)
            C2 = coords[1]
            N3 = coords[2]
            C4 = coords[3]
            C5 = coords[4]
            
            v1 = C2 - N3
            v2 = C4 - N3
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            # 计算当前C5-N3向量
            v3 = C5 - N3
            v3 = v3 / np.linalg.norm(v3)
            
            current_angle = np.arcsin(np.abs(np.dot(v3, normal)))            # 计算当前out-of-plane角度
            # print(f"当前out-of-plane角度: {np.degrees(current_angle):.2f}度")
            
            rotation_axis = np.cross(v3, normal)  # 注意这里交换了顺序              # 计算旋转轴（C5-N3向量与平面法向量的叉积）
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            
            rotation_angle = target_angle - current_angle            # 计算需要旋转的角度
            # print(f"需要旋转的角度: {np.degrees(rotation_angle):.2f}度")  
            
            c = np.cos(rotation_angle)            # 构建旋转矩阵
            s = np.sin(rotation_angle)
            t = 1 - c
            x, y, z = rotation_axis
            
            rotation_matrix = np.array([
                [t*x*x + c,    t*x*y - s*z,  t*x*z + s*y],
                [t*x*y + s*z,  t*y*y + c,    t*y*z - s*x],
                [t*x*z - s*y,  t*y*z + s*x,  t*z*z + c]
            ])
            
            v3 = C5 - N3            # 旋转C5原子
            v3_rotated = np.dot(rotation_matrix, v3)
            coords[4] = N3 + v3_rotated
            
            # v3_new = coords[4] - coords[2]            # 验证旋转后的距离
            # distance_before = np.linalg.norm(v3)
            # distance_after = np.linalg.norm(v3_new)
            # # print(f"旋转前距离: {distance_before:.3f} Å，旋转后距离: {distance_after:.3f} Å")
            
            # v3_new = v3_new / np.linalg.norm(v3_new)            # 验证旋转后的角度
            # final_angle = np.arcsin(np.dot(v3_new, normal))
            # # print(f"旋转后out-of-plane角度: {np.degrees(final_angle):.2f}度")
        
        distances = pdist(coords)            # 计算21个原子对距离
        
        C2 = coords[1]
        N3 = coords[2]
        C4 = coords[3]
        C5 = coords[4]
        
        v1 = C2 - N3
        v2 = C4 - N3
        normal = np.cross(v1, v2)
        normal = normal / np.linalg.norm(normal)
        
        v3 = C5 - N3
        v3 = v3 / np.linalg.norm(v3)
        
        angle = np.arcsin(np.abs(np.dot(v3, normal)))
        sin_angle = np.sin(angle)
        cos_angle = np.cos(angle)
        
        # 组合标签：21个距离 + sin(angle) + cos(angle)
        label = np.concatenate([distances, [sin_angle, cos_angle]])
        
        coords_list.append(coords)
        labels_list.append(label)
    
    coords_array = np.array(coords_list)
    labels_array = np.array(labels_list)
    
    np.savez('/Users/jinc_air/Documents/ML_UED/NMM/nmm_structure10000.npz', labels=labels_array, coords=coords_array)
    print(f'Generated {len(labels_array)} NMM structure labels.')
    
    return coords_array, labels_array

def save_to_h5(scattering, labels, batch_index, h5_file):
    with h5py.File(h5_file, 'a') as h5f:
        # 如果文件不存在，创建组
        if 'scattering' not in h5f:
            h5f.create_group('scattering')
        if 'labels' not in h5f:
            h5f.create_group('labels')
        
        # 保存数据
        h5f['scattering'].create_dataset(f'batch_{batch_index}', data=scattering)
        h5f['labels'].create_dataset(f'batch_{batch_index}', data=labels)

def compute_scattering_signals_NMM(labels, batch_size, sM_ground, s, ScatAmp, atom_pairs):
    """
    计算一维散射信号并存储为不同batch文件,同时保存标签
    :param labels: 输入的标签数组 (n_samples, 23) 包含21个原子对距离和2个角度参数
    :param batch_size: 每个batch的大小
    :param sM_ground: 基态信号
    :param s: 距离数组
    :param ScatAmp: 散射振幅数组
    :param atom_pairs: 原子对索引列表
    """
    start_time = time.time()
    num_batches = (len(labels) + batch_size - 1) // batch_size

    total_mean = 0.0
    total_std_dev = 0.0
    total_count = 0

    total_mean_labels = np.zeros(23)  # 21个原子对距离 + 2个角度参数
    total_std_dev_labels = np.zeros(23)
    total_count_labels = 0

    for i in range(num_batches):
        print(f'Processing batch {i}, with {num_batches} number batches.')
        batch_labels = labels[i * batch_size:(i + 1) * batch_size]  # 获得分子的标签
        batch_scattering = []
        batch_labels_processed = []

        print(f'the shape of batch_labels is {batch_labels.shape}')
        with Pool(processes=7) as pool:
            results = pool.starmap(calculate_point_NMM, [(label, s, ScatAmp, sM_ground, atom_pairs) for label in batch_labels])

        for delpattern_noisy, label in results:
            batch_scattering.append(delpattern_noisy)
            batch_labels_processed.append(label)

        # 直接使用 batch_scattering 进行展平操作
        batch_patterns_flattened = np.concatenate([delpattern_noisy.flatten() for delpattern_noisy in batch_scattering])
        batch_mean = np.mean(batch_patterns_flattened)
        batch_std_dev = np.std(batch_patterns_flattened)
        batch_count = len(batch_patterns_flattened)

        # 更新标签的平均值和标准差
        batch_labels_mean = np.mean(batch_labels_processed, axis=0)
        batch_labels_std_dev = np.std(batch_labels_processed, axis=0)
        label_count = len(batch_labels_processed)

        # 更新总的平均值和标准差
        new_total_count = total_count + batch_count
        new_total_mean = (total_mean * total_count + batch_mean * batch_count) / new_total_count
        new_total_std_dev = np.sqrt((total_count * (total_std_dev ** 2 + (total_mean - new_total_mean) ** 2) + 
                                 batch_count * (batch_std_dev ** 2 + (batch_mean - new_total_mean) ** 2)) / new_total_count)
        
        # 更新总的标签平均值和标准差
        new_total_count_labels = total_count_labels + label_count
        new_total_mean_labels = (total_mean_labels * total_count_labels + batch_labels_mean * label_count) / new_total_count_labels
        new_total_std_dev_labels = np.sqrt((total_count_labels * (total_std_dev_labels ** 2 + (total_mean_labels - new_total_mean_labels) ** 2) + 
                                    label_count * (batch_labels_std_dev ** 2 + (batch_labels_mean - new_total_mean_labels) ** 2)) / new_total_count_labels)

        total_mean = new_total_mean
        total_std_dev = new_total_std_dev
        total_count = new_total_count

        total_mean_labels = new_total_mean_labels
        total_std_dev_labels = new_total_std_dev_labels
        total_count_labels = new_total_count_labels

        print(f'the total mean is {total_mean}, the totalstd_dev is {total_std_dev}, the totalcount is {total_count}')
        print(f'the total mean_labels is {total_mean_labels}, the totalstd_dev_labels is {total_std_dev_labels}, the totalcount_labels is {total_count_labels}')
        print(f'Batch {i} processing completed. timestamp: {time.time() - start_time:.2f}')
        np.savez(f'./train_set/training_batch_{i+4}.npz', scattering=batch_scattering, labels=batch_labels_processed)

        # 删除不再需要的变量以释放内存
        del batch_labels
        del batch_scattering
        del batch_labels_processed
        del batch_patterns_flattened
        gc.collect()  # 强制进行垃圾回收

    end_time = time.time()
    print(f'Total computation time: {end_time - start_time:.2f} seconds')

    # 保存统计信息到json文件
    stats = {
        'total_mean': float(total_mean),
        'total_std_dev': float(total_std_dev),
        'total_count': total_count,
        'total_mean_labels': total_mean_labels.tolist(),
        'total_std_dev_labels': total_std_dev_labels.tolist(),
        'total_count_labels': total_count_labels
    }
    with open('statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
        print(f'Saved statistics to file statistics.json')

    return total_count_labels

def calculate_point_NMM(label, s, ScatAmp, sM_ground, atom_pairs):
    """
    计算单个NMM分子的衍射信号
    :param label: 23个标签（21个原子对距离 + sin(angle) + cos(angle)）
    :param s: 距离数组
    :param ScatAmp: 散射振幅数组
    :param sM_ground: 基态信号
    :param atom_pairs: 原子对索引列表
    :return: 处理后的衍射信号和标签
    """
    # 计算IA（原子散射振幅平方和）
    SimIA = np.sum(np.abs(ScatAmp.T)**2, axis=1)
    SimIA = SimIA.squeeze()
    
    # 计算IM（干涉项）
    SimIM = np.zeros_like(s, dtype=np.complex128)
    
    # 计算所有原子对的干涉项（只使用前21个距离值）
    for idx, r in enumerate(label[:21]):  # 只使用前21个距离值
        i, j = atom_pairs[idx]
        SimIM += ScatAmp[i] * np.conj(ScatAmp[j]).T * np.sin(s * r) / (s * r)
    
    SimIM = 2 * np.real(SimIM)
    SimIM = SimIM * 1e14
    SimIA = SimIA * 1e14

    # 添加随机噪声，噪声级别在万分之一到千分之一之间 对应实验数据count ~ 1量级
    noise_level = np.random.uniform(1e-2, 1)
    noise = np.random.normal(0, noise_level*s**-1, SimIM.shape)
    SimIM_noisy = SimIM + noise
    del_sM = s * SimIM_noisy / SimIA  # 计算delta sM 信号
    del_sM[s < 1] = 0
    del_sM[s > 12] = 0
    #     # 绘制SimIM图像
    # plt.figure()
    # plt.plot(s, del_sM)
    # plt.title('SimIM vs s')
    # plt.xlabel('s')
    # plt.ylabel('SimIM')
    # plt.grid(True)
    # plt.show()

    # 生成2D pattern
    X, Y = np.meshgrid(np.arange(-210, 211), np.arange(-210, 211))
    y, x = X.shape
    Radius = np.sqrt(X**2 + Y**2) * 0.05
    PatternSynthesized = np.zeros_like(X, dtype=np.float64)

    for i in range(y):
        for j in range(x):
            radius = Radius[i, j]
            if radius > 0:
                radius_index = round(radius / 0.0245)
                PatternSynthesized[i, j] = del_sM[radius_index - 1] - sM_ground[i,j]

    #     # 绘制2D pattern
    # plt.figure()
    # plt.pcolormesh(PatternSynthesized, shading='gouraud', cmap='jet', vmin=-4, vmax=4)
    # plt.colorbar()
    # plt.title('Ground state pattern')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('equal')
    # plt.show()

    delpattern_noisy = PatternSynthesized.astype(np.float32)

    return delpattern_noisy, label

def compute_NMM_ground_signals(coords, ScatAmp, s, atom_pairs):
    """
    计算NMM分子的基态散射信号
    :param coords: 分子坐标 (7,3)
    :param ScatAmp: 散射振幅数组
    :param s: 距离数组
    """

    distances = pdist(coords)   
    print(f'the ground distances pairs are {distances}')
    
    # 计算IA（原子散射振幅平方和）
    SimIA = np.sum(np.abs(ScatAmp.T)**2, axis=1)
    SimIA = SimIA.squeeze()
    
    # 计算IM（干涉项）
    SimIM = np.zeros_like(s, dtype=np.complex128)
    # 计算所有原子对的干涉项
    for (i, j), r in zip(atom_pairs, distances):
        SimIM += ScatAmp[i] * np.conj(ScatAmp[j]).T * np.sin(s * r) / (s * r)
    
    SimIM = 2 * np.real(SimIM)
    del_sM = s * SimIM / SimIA  # 计算delta sM 信号
    del_sM[s < 1] = 0
    del_sM[s > 12] = 0

    # 绘制SimIM图像
    plt.figure()
    plt.plot(s, del_sM)
    plt.title('SimIM vs s')
    plt.xlabel('s')
    plt.ylabel('SimIM')
    plt.grid(True)
    plt.show()

    # 生成2D pattern
    X, Y = np.meshgrid(np.arange(-210, 211), np.arange(-210, 211))
    y, x = X.shape
    Radius = np.sqrt(X**2 + Y**2) * 0.05
    PatternSynthesized = np.zeros_like(X, dtype=np.float64)

    for i in range(y):
        for j in range(x):
            radius = Radius[i, j]
            if radius > 0:
                radius_index = round(radius / 0.0245)
                PatternSynthesized[i, j] = del_sM[radius_index - 1]

    # 绘制2D pattern
    plt.figure()
    plt.pcolormesh(PatternSynthesized, shading='gouraud', cmap='jet', vmin=-5, vmax=5)
    plt.colorbar()
    plt.title('Ground state pattern')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('equal')
    plt.show()

    # 保存基态信号
    np.save('./NMM_ground_sM.npy', PatternSynthesized)
    print('Ground state signal computed and saved.')


def generate_2d_patterns_from_NMM_s0(s, fC, fN, fO, s0_files):
    """
    从多个s0矩阵生成二维图，并计算每个位置的均值和标准差
    :param s0_files: 包含多个s0矩阵文件路径的列表
    :param s: 距离数组
    :param fC: C原子的散射振幅
    :param fN: N原子的散射振幅
    :param fO: O原子的散射振幅
    """
    n_patterns_per_file = 45
    total_patterns = n_patterns_per_file * len(s0_files)
    X, Y = np.meshgrid(np.arange(-210, 211), np.arange(-210, 211))
    Radius = np.sqrt(X**2 + Y**2) * 0.05
    
    # 修改IA的计算逻辑：按照NMM分子中C/N/O原子的数量计算
    # NMM分子中C原子数量为5，N原子数量为1，O原子数量为1
    SimIA = np.abs(fC.T)**2 * 5 + np.abs(fN.T)**2 + np.abs(fO.T)**2
    SimIA = 1e14 * SimIA.squeeze()

    # 计算NMM基态标签
    coords, _ = read_xyz_extract_cno("NMM_eq.xyz")
    distances = pdist(coords)  # 计算21个原子对距离
    
    # 计算out-of-plane角度
    C2 = coords[1]
    N3 = coords[2]
    C4 = coords[3]
    C5 = coords[4]
    
    v1 = C2 - N3
    v2 = C4 - N3
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    v3 = C5 - N3
    v3 = v3 / np.linalg.norm(v3)
    
    angle = np.arcsin(np.dot(v3, normal))
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)
    
    # 组合标签：21个距离 + sin(angle) + cos(angle)
    ground_label = np.concatenate([distances, [sin_angle, cos_angle]])
    
    # 优化内存使用：按需扩展数组而非预先分配
    all_patterns = np.zeros((len(s0_files), n_patterns_per_file, X.shape[0], X.shape[1]), dtype=np.float32)
    all_labels = np.array([ground_label] * total_patterns)  # 使用计算得到的基态标签

    for file_idx, s0_file in enumerate(s0_files):
        s0 = sio.loadmat(s0_file)['s0']
        print(f'Processing {s0_file}...')
        
        for pattern_idx_in_file in range(n_patterns_per_file):
            # 确保s长度与s0维度匹配
            current_s0 = s0[pattern_idx_in_file]
            assert len(s) == len(current_s0), f"s length {len(s)} mismatch with s0 pattern {len(current_s0)}"
            
            del_sM = current_s0 * s / SimIA  
            del_sM[(s < 1.5) | (s > 12)] = 0  # 修改s的范围限制
            del_sM /= 22  # 归一化因子

            # 向量化计算PatternSynthesized
            radius_flat = Radius.flatten()
            radius_indices = np.round(radius_flat / 0.0245).astype(int)
            radius_indices[radius_flat <= 0] = 0
            radius_indices = np.clip(radius_indices, 1, len(del_sM)-1)  # 防止越界
            
            pattern_flat = del_sM[radius_indices - 1]
            PatternSynthesized = pattern_flat.reshape(X.shape).astype(np.float32)

            # # 添加随机噪声
            # noise_level = np.random.uniform(1e-2, 1)
            # noise = np.random.normal(0, noise_level, PatternSynthesized.shape)
            # PatternSynthesized += noise

            # delpattern_noisy = PatternSynthesized.astype(np.float32)


            all_patterns[file_idx, pattern_idx_in_file] = PatternSynthesized

            # # 绘制2D pattern
            # plt.figure()
            # plt.pcolormesh(PatternSynthesized, shading='gouraud', cmap='jet', vmin=-5, vmax=5)
            # plt.colorbar()
            # plt.title('Ground state pattern')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # plt.axis('equal')
            # plt.show()
            # 释放临时变量
            gc.collect()

    # 后续处理保持不变
    # all_patterns = np.clip(all_patterns, -10, 10)
    np.savez('pattern_experiments200_NMM.npz', scattering=all_patterns, labels=all_labels)

def calculate_dataset_statistics():
    """
    读取所有NMM训练集文件，计算所有数据的均值和标准差
    统计内容包括：
    1. 21个原子对距离的均值和标准差（主要训练目标）
    2. 2个角度参数（sin和cos）的均值和标准差（辅助训练目标）
    3. 散射信号的均值和标准差
    """
    import glob
    import os

    total_mean = 0.0
    total_std_dev = 0.0
    total_count = 0
    total_mean_labels_main = np.zeros(21)  # 21个原子对距离（主要训练目标）
    total_std_dev_labels_main = np.zeros(21)
    total_mean_labels_aux = np.zeros(2)  # 2个角度参数（辅助训练目标）
    total_std_dev_labels_aux = np.zeros(2)
    total_count_labels = 0

    # 获取所有训练集文件
    batch_files = glob.glob('./train_set/training_batch_*.npz')
    for batch_file in batch_files:
        print(f'Processing {batch_file}...')
        data = np.load(batch_file)
        batch_scattering = data['scattering']
        batch_labels = data['labels']

        # 计算散射信号的统计量
        batch_patterns_flattened = np.concatenate([pattern.flatten() for pattern in batch_scattering])
        batch_mean = np.mean(batch_patterns_flattened)
        batch_std_dev = np.std(batch_patterns_flattened)
        batch_count = len(batch_patterns_flattened)

        # 计算标签的统计量（分为主要和辅助）
        batch_labels_main = batch_labels[:, :21]  # 前21个原子对距离
        batch_labels_aux = batch_labels[:, 21:]   # 后2个角度参数
        
        batch_labels_main_mean = np.mean(batch_labels_main, axis=0)
        batch_labels_main_std_dev = np.std(batch_labels_main, axis=0)
        batch_labels_aux_mean = np.mean(batch_labels_aux, axis=0)
        batch_labels_aux_std_dev = np.std(batch_labels_aux, axis=0)
        label_count = len(batch_labels)

        # 更新总的平均值和标准差
        new_total_count = total_count + batch_count
        new_total_mean = (total_mean * total_count + batch_mean * batch_count) / new_total_count
        new_total_std_dev = np.sqrt((total_count * (total_std_dev ** 2 + (total_mean - new_total_mean) ** 2) + 
                                 batch_count * (batch_std_dev ** 2 + (batch_mean - new_total_mean) ** 2)) / new_total_count)

        # 更新主要标签的统计量
        new_total_count_labels = total_count_labels + label_count
        new_total_mean_labels_main = (total_mean_labels_main * total_count_labels + batch_labels_main_mean * label_count) / new_total_count_labels
        new_total_std_dev_labels_main = np.sqrt((total_count_labels * (total_std_dev_labels_main ** 2 + (total_mean_labels_main - new_total_mean_labels_main) ** 2) + 
                                    label_count * (batch_labels_main_std_dev ** 2 + (batch_labels_main_mean - new_total_mean_labels_main) ** 2)) / new_total_count_labels)
        
        # 更新辅助标签的统计量
        new_total_mean_labels_aux = (total_mean_labels_aux * total_count_labels + batch_labels_aux_mean * label_count) / new_total_count_labels
        new_total_std_dev_labels_aux = np.sqrt((total_count_labels * (total_std_dev_labels_aux ** 2 + (total_mean_labels_aux - new_total_mean_labels_aux) ** 2) + 
                                    label_count * (batch_labels_aux_std_dev ** 2 + (batch_labels_aux_mean - new_total_mean_labels_aux) ** 2)) / new_total_count_labels)

        total_mean = new_total_mean
        total_std_dev = new_total_std_dev
        total_count = new_total_count

        total_mean_labels_main = new_total_mean_labels_main
        total_std_dev_labels_main = new_total_std_dev_labels_main
        total_mean_labels_aux = new_total_mean_labels_aux
        total_std_dev_labels_aux = new_total_std_dev_labels_aux
        total_count_labels = new_total_count_labels

        print(f'Processed {batch_file}: {label_count} samples, mean={total_mean:.4f}, std={total_std_dev:.4f}')

    print(f'\nFinal: {total_count_labels} samples, mean={total_mean:.4f}, std={total_std_dev:.4f}')
  
    # 保存统计信息到json文件
    stats = {
        'total_mean': float(total_mean),
        'total_std_dev': float(total_std_dev),
        'total_count': total_count,
        'total_mean_labels': total_mean_labels_main.tolist(),  # 使用main标签的均值
        'total_std_dev_labels': total_std_dev_labels_main.tolist(),  # 使用main标签的标准差
        'total_count_labels': total_count_labels
    }
    with open('statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
        print(f'Saved dataset statistics to file statistics.json')

def read_xyz_extract_cno(filename):
    """
    读取xyz文件，只提取C/N/O原子。 返回:coords: shape=(7,3) elements: ['C', 'C', ..., 'O']
    """
    with open(filename, 'r') as f:
        lines = f.readlines()
    elements = []
    coords = []
    for line in lines[2:]:  # 跳过前两行（原子数和注释）
        parts = line.strip().split()
        if len(parts) == 4 and parts[0] in ['C', 'N', 'O']:
            elements.append(parts[0])
            coords.append([float(x) for x in parts[1:]])
    return np.array(coords), elements


if __name__ == "__main__":
    # 加载初始结构
    coords, elements = read_xyz_extract_cno("./coords/NMM_eq.xyz")
    # coords, elements = read_xyz_extract_cno("NMM_pl_reordered.xyz")
    # coords, elements = read_xyz_extract_cno("NMM_ax_reordered.xyz")
    # print(f'the coords and elements are: {coords},{elements}')
    print(f'the coords and elements are: {coords},{elements}')

    distances = pdist(coords)  # 计算21个原子对距离
    print(f'the distances are: {distances}')
    n_atoms = len(coords)
    atom_pairs = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            atom_pairs.append((i, j))

    # 使用输出重定向：python train_gen.py | tee output.txt
    # 加载fC和fS  fH fO
    fC = sio.loadmat('../DPWA/fC.mat')['fC']
    fN = sio.loadmat('../DPWA/fN.mat')['fN']
    fO = sio.loadmat('../DPWA/fO.mat')['fO']
    # 按照elements的原子顺序构建散射振幅   ['C', 'C', 'N', 'C', 'C', 'C', 'O']
    ScatAmp = np.concatenate((fC, fC, fN, fC, fC, fC, fO))  # 这里按照分子坐标的顺序对应散射振幅

    # 定义距离数组
    RadiusRange = np.arange(1, 681)
    sPixel = 0.0245
    s = RadiusRange * sPixel - sPixel / 2

    # # 计算并保存基态信号
    # print('Computing ground state signal...')
    # # compute_NMM_ground_signals(coords, ScatAmp, s, atom_pairs)
    # sM_ground = np.load('./NMM_ground_sM.npy')  # 加载基态信号
    
    # # 生成扰动结构
    # coords, labels = generate_nmm_structure_variation(coords, 10000, 0.2,0.01)

    # #  同时包含两种变化
    # # coords, labels = generate_NMM_structure(coords, 10000, 0.1, (-60, 60))
    # print(f'the length of coords and labels are: {len(coords)},{len(labels)}, and the shape of coords and labels are: {coords.shape},{labels.shape}')
    
    # # 计算散射信号
    # batch_size = 2500
    # total_count_labels = compute_scattering_signals_NMM(labels, batch_size, sM_ground, s, ScatAmp, atom_pairs)
    # print(f'Total number of events: {total_count_labels}')
    # print('scattering signals computed.')

    # 生成2D patterns from s0
    # s0_files = [f'/Users/jinc_air/Documents/ML_UED/NMM/bootstrapping2/s0_{i}.mat' for i in range(200)]  # 假设有100个s0文件
    # generate_2d_patterns_from_NMM_s0(s[0:637], fC[:,0:637], fN[:,0:637], fO[:,0:637], s0_files)
    # print('2D patterns generated from s0 matrices.')

    # calculate_dataset_statistics()


