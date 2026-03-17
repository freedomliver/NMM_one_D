import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split, Dataset
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from torch.cuda.amp import autocast, GradScaler
import json
import csv
import h5py
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR
from torch.optim.lr_scheduler import ChainedScheduler

# 定义 ResNet50 的卷积块
class ConvolutionalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(ConvolutionalBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels[2])
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

# 定义 ResNet50 的恒等块
class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IdentityBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += identity
        out = self.relu(out)
        return out

# 构建 ResNet50 模型
class ResNet50(nn.Module):
    def __init__(self, input_channels=1, output_dim=3):
        super(ResNet50, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            ConvolutionalBlock(64, [64, 64, 256], stride=1),
            IdentityBlock(256, [64, 64, 256]),
            IdentityBlock(256, [64, 64, 256])
        )

        self.layer2 = nn.Sequential(
            ConvolutionalBlock(256, [128, 128, 512], stride=2),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512]),
            IdentityBlock(512, [128, 128, 512])
        )

        self.layer3 = nn.Sequential(
            ConvolutionalBlock(512, [256, 256, 1024], stride=2),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024]),
            IdentityBlock(1024, [256, 256, 1024])
        )

        self.layer4 = nn.Sequential(
            ConvolutionalBlock(1024, [512, 512, 2048], stride=2),
            IdentityBlock(2048, [512, 512, 2048]),
            IdentityBlock(2048, [512, 512, 2048])
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x

# 定义 ResNet18 的基本块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(identity)
        out = self.relu(out)
        return out

# 构建 ResNet18 模型
class ResNet18(nn.Module):
    def __init__(self, input_channels=1, output_dim=3):
        super(ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet18使用BasicBlock而不是Bottleneck
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128)
        )
        
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256, stride=2),
            BasicBlock(256, 256)
        )
        
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512, stride=2),
            BasicBlock(512, 512)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.fc(torch.flatten(self.avgpool(x), 1))
        return x

def custom_multi_task_loss(output, target, main_indices=list(range(21)), aux_indices=[21,22], w_main=1.0, w_aux=0.2):
    mse = nn.MSELoss()
    loss_main = mse(output[:, main_indices], target[:, main_indices])
    loss_aux = mse(output[:, aux_indices], target[:, aux_indices])
    return w_main * loss_main + w_aux * loss_aux

class NMMDataset(Dataset):
    def __init__(self, h5_file, indices, mean, std, mean_labels, std_labels):
        self.h5_file = h5_file
        self.indices = indices
        self.mean = mean
        self.std = std
        self.mean_labels = mean_labels
        self.std_labels = std_labels
        
        # 计算每个batch的大小
        with h5py.File(h5_file, 'r') as h5f:
            self.batch_sizes = [len(h5f['scattering'][f'batch_{i}']) for i in range(len(h5f['scattering']))]
            self.cumulative_sizes = np.cumsum(self.batch_sizes)
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        
        # 找到对应的batch
        batch_idx = np.searchsorted(self.cumulative_sizes, global_idx, side='right')
        if batch_idx > 0:
            local_idx = global_idx - self.cumulative_sizes[batch_idx - 1]
        else:
            local_idx = global_idx
            
        with h5py.File(self.h5_file, 'r') as h5f:
            inputs = torch.tensor(h5f['scattering'][f'batch_{batch_idx}'][local_idx], dtype=torch.float32).unsqueeze(0)
            labels = torch.tensor(h5f['labels'][f'batch_{batch_idx}'][local_idx], dtype=torch.float32)
            
        # 标准化
        inputs = (inputs - self.mean) / self.std
        labels = (labels - self.mean_labels) / self.std_labels
        
        return inputs, labels

def load_data(h5_file, train_ratio=0.8):
    # 计算总样本数
    total_samples = 0
    with h5py.File(h5_file, 'r') as h5f:
        for i in range(len(h5f['scattering'])):
            total_samples += len(h5f['scattering'][f'batch_{i}'])
    
    # 创建索引列表并划分训练集和验证集
    indices = list(range(total_samples))
    np.random.shuffle(indices)
    num_train = int(train_ratio * total_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    return train_indices, val_indices

# 添加读取statistics.json文件的函数
def load_statistics(file_path='statistics.json'):
    with open(file_path, 'r') as f:
        stats = json.load(f)
    return stats['total_mean'], stats['total_std_dev'], stats['total_mean_labels'], stats['total_std_dev_labels']
# 添加数据标准化函数
def standardize(inputs, mean, std):
    return (inputs - mean) / std

def train_resnet(model, h5_file, epochs=10, batch_size=100, device='cpu', patience=7):
    # 读取标准化参数
    mean, std, mean_labels, std_labels = load_statistics()
    mean = torch.tensor(mean)
    std = torch.tensor(std)
    mean_labels = torch.tensor(mean_labels)
    std_labels = torch.tensor(std_labels)

    model.to(device)
    criterion = custom_multi_task_loss
    
    # 设置初始学习率
    initial_lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=0.01)
    
    # 设置预热阶段
    warmup_epochs = 5
    warmup_scheduler = LinearLR(optimizer, 
                              start_factor=0.1, 
                              end_factor=1.0, 
                              total_iters=warmup_epochs)
    
    # 设置余弦退火调度器，带有周期性重启
    cosine_scheduler = CosineAnnealingWarmRestarts(optimizer, 
                                                  T_0=10,  # 第一次重启的周期
                                                  T_mult=2,  # 每次重启后周期翻倍
                                                  eta_min=1e-6)  # 最小学习率
    
    # 链接预热和余弦退火调度器
    scheduler = ChainedScheduler([warmup_scheduler, cosine_scheduler])

    loss_collector = []
    val_loss_collector = []
    lr_collector = []  # 收集学习率变化
    scaler = GradScaler()

    # 获取训练集和验证集的索引
    train_indices, val_indices = load_data(h5_file)
    
    # 创建数据集
    train_dataset = NMMDataset(h5_file, train_indices, mean, std, mean_labels, std_labels)
    val_dataset = NMMDataset(h5_file, val_indices, mean, std, mean_labels, std_labels)
    
    print('数据集准备完成')
    
    # 检查是否有保存的模型参数和优化器状态
    if os.path.exists('resnet50_model_NMM.pth'):
        model.load_state_dict(torch.load('resnet50_model_NMM.pth', map_location=device))
        print("已加载已保存模型参数")
    if os.path.exists('optimizer_state_NMM.pth'):
        optimizer.load_state_dict(torch.load('optimizer_state_NMM.pth', map_location=device))
        print("已加载已保存优化器状态")

    torch.cuda.empty_cache()
    print(f"Allocated memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    print(f"Reserved memory : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
    
    # 读取之前的验证损失
    best_val_loss = float('inf')
    if os.path.exists('loss_values.csv'):
        with open('loss_values.csv', mode='r', newline='') as file:
            reader = csv.reader(file)
            next(reader, None)
            for row in reader:
                best_val_loss = min(best_val_loss, float(row[2]))
    print(f"初始最佳验证损失: {best_val_loss}")

    # 添加早停相关变量
    epochs_no_improve = 0

    # 添加文件操作，打开CSV文件用于记录损失值
    with open('loss_values.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Learning Rate'])

        for epoch in range(epochs):
            print(f'Epoch {epoch+1}/{epochs}')
            epoch_loss = 0
            model.train()

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                
                # 在每个batch后更新学习率
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                lr_collector.append(current_lr)
                
                if i % 10 == 0:  # 每10个batch打印一次
                    print(f'Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}, LR: {current_lr:.6f}')

            epoch_loss /= len(train_loader)
            loss_collector.append(epoch_loss)
            print(f'Epoch [{epoch+1}/{epochs}], Training Loss: {epoch_loss:.4f}')

            # 验证过程
            model.eval()
            val_epoch_loss = 0
            with torch.no_grad():
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)

                    with autocast():
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    val_epoch_loss += loss.item()
            
            val_epoch_loss /= len(val_loader)
            val_loss_collector.append(val_epoch_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1}/{epochs}], Validation Loss: {val_epoch_loss:.5f}, learning rate: {current_lr:.6f}')

            writer.writerow([epoch+1, epoch_loss, val_epoch_loss, current_lr])

            if val_epoch_loss < best_val_loss:
                best_val_loss = val_epoch_loss
                epochs_no_improve = 0
                torch.save(model.state_dict(), 'resnet50_model_NMM.pth')
                torch.save(optimizer.state_dict(), 'optimizer_state_NMM.pth')
                print("模型和优化器状态已保存")
            else:
                epochs_no_improve += 1
                if epochs_no_improve == patience:
                    print(f"验证损失在 {patience} 个epoch内没有改善，提前停止训练")
                    break

            torch.cuda.empty_cache()

    # 绘制损失变化图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # 绘制损失曲线
    ax1.plot(loss_collector, label='Training Loss')
    ax1.plot(val_loss_collector, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss over Epochs')
    ax1.legend()
    
    # 绘制学习率变化曲线
    ax2.plot(lr_collector, label='Learning Rate')
    ax2.set_xlabel('Batch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.set_yscale('log')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')

if __name__ == '__main__':
    # 加载数据集    
    print('加载数据集')
    h5_file = 'training_data.h5'

    # 构建 ResNet50 模型
    # model = ResNet50(input_channels=1, output_dim=23)
    # 构建 ResNet18 模型
    model = ResNet18(input_channels=1, output_dim=23)
    # 将模型移动到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('cuda.cache emptied')
    print(device)
    model.to(device)

    # 训练模型
    print("开始训练模型...")
    train_resnet(model, h5_file, epochs=40, batch_size=128, device=device, patience=1)
