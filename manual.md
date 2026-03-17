1> 项目简介：
这是一个使用机器学习分析UED数据的项目。实现从径向积分的一维衍射信号预测分子结构的特征信息。
UED是超快电子衍射装置，实验的数据是二维图像。UED采用泵浦探测实验方式，即每轮实验都有一个泵浦光作用的时间零点，通过时间零点后数据与时间零点前的数据差分得到反应过程的动力学信息。数据的存储格式通常是，一个文件夹里面存图片分别对应不同的时刻，不同文件夹对应不同的重复实验。对实验数据的预处理已经有现成的Matlab程序，可以读取并处理图像后先得到平均的图像（按实验组数平均），这是三维矩阵int（pattern_size1,pattern_size2,num_points），接着减去时间零点前的均值（差分），然后对差分矩阵按照离中心距离进行径向积分降维，得到二维矩阵int（s,num_points），这里的s是电子散射的转移动量大小, num_points对应了不同时刻的信号。
本项目分析的分子是NMM分子，config文件给了基态的分子结构坐标。
本项目用训练得到的神经网络对实验数据进行回归预测。这里把二维矩阵int（s,num_points）的不同num_points作为不同的数据，即神经网络的输入数据格式是一维数据int(s)。即不考虑不同num_points之间的时许关系，单独进行回归预测分析。网络的输出结果是几个特征原子的全距离矩阵。具体参考生成训练集的说明。

2> 本项目的文件结构：
根目录下：manual.md（本文件，请后续保留原本的说明内容，请在现有说明内容后面添加使用指南）
    config.py：配置文件
    python项目的环境配置
    requirements.txt：项目依赖包版本信息
    /params：文件夹，用来存储项目运行中需要存储的参数
    /file_training_data：文件夹，用来存储生成训练数据
    train_gen_NMM.py：生成NMM分子的训练数据。
    functions.py：定义一些函数，神经网络。如数据处理函数，数据保存函数，数据归一化函数等。
    train.py：训练神经网络
    inference.py：测试神经网络,预测实验数据结果


3> 详细说明每个文件是什么，做什么的：
1 manual.md：使用说明文件，请后续保留原本的说明内容，请在现有说明内容后面添加使用指南。
2 config.py：项目配置文件，在合适位置说明项目环境配置。
本工作电脑是Mac air3 M3 16G+256G. 已经安装了homebrew,miniconda，本项目的环境配置在当前目录下。
我比较习惯使用pytorch，请根据项目要求选择对应的pytorch、python、numpy等包的版本。
3 requirements.txt：项目依赖包版本信息，请根据项目要求安装对应的包版本。如有变动及时更新对应的版本信息。
4 train_gen_NMM.py：生成NMM分子的训练数据。这部分内容可以参考reference/train50_gen_NMM.py,但是这个文件很多地方不对，需要你重新写。
        一、旧文件只考虑了C，N，O原子的散射振幅，这里计算散射信号时需要计算H原子的散射振幅。H原子参与干涉项计算，使用相对固定位置，即C1- C4的H原子相对C1- C4固定，N上的甲基C5 的H相对C5固定；不考虑同一个C原子上H原子旋转拉伸等变化。就是每个H和对应的C完全锁死。
        二、旧文件使用二维的图片作为神经网络的数据数据，这里不使用这么大的二维图片。只使用一维的信号作为输入信号。
        三、旧文件因为二维图片太大而使用421*421尺寸，这里使用一维信号就不进行缩减，信号长度保持681.即输入为一维长度681.对应s数组的长度。
        四、旧文件考虑了CNO共7个原子的空间分布，这里认为O和平面的四个C保持不动，即只用考虑N基的几个原子移动，注意这里不考虑H原子的移动仅考虑散射信号。H原子默认相对固定，即平面四个C的H相对对应的C不变，N基上的C的H相对于这个C不变。
        自由度：O和平面四个C不动，只考虑N基的几个原子移动，可以O和4个C坐标锁死，剩下的自由度有N-0键长，C2- N键长（需要多一个C来锁定N的位置），限定N原子只能在中轴面上变化位置，N- C5，O- C5，C2- C5 键长。
        五、根据上面的这些自由度请你确定如何生成训练数据。（理论和实验大致表明整个反应过程是N基的平面化振动，即N和N上的甲基在一定区域内变化）。初步的训练数据量在100万条左右。
        六、训练信号需要噪声，请根据实际情况选择最合适的噪声添加方式，可以添加多种噪声。不要求使用相同方式，只参考噪声的数量级大小，可根据需要叠加不同类型噪声。
        七、因为一维数据大小很小，计算信号时不用并行计算，正常计算即可，（添加恰当的耗时输出）
    训练数据的标签以有自由度原子的全距离矩阵为标签，即O原子，N原子，N上的C原子,即只考虑O，N，C5。剩下4个C原子与O原子相对固定。
    数据保存按照.h5文件保存在file_training_data中；为了在训练模型时进行归一化，请保存归一化参数，保存在params文件夹中（继续使用json格式保存）。不同数据集命名上要有所区别。
5 params：文件夹，用来存储项目运行中需要存储的参数，包括分子坐标，散射振幅等。
6 file_training_data：文件夹，用来存储生成训练数据
7 functions.py：定义一些函数，神经网络。如数据处理函数，数据保存函数，数据归一化函数等。
        根据本项目实际需要，请你确定使用那种神经网络比较合适。本项目是一维数据回归，神经网络要具备倒空间反演能力（即傅立叶变换能力）
        神经网络使用class类进行定义。
        除神经网络外，使用到的其它函数，特别是类似全局函数，可以一并都在此脚本中定义。比如归一化函数等。
8 train.py：训练神经网络
        将训练数据集随机分成训练集和测试集，参考reference/train50_gen_NMM.py.设置合适的学习率，比如合适的退火等
        本项目的网络训练计划在本机进行，所以每个batch 128。
9 inference.py：测试神经网络,预测实验数据结果
        推理分两种，一种是调用train_gen_NMM.py的方式生成10000条具有特征的计算数据，然后进行推理。分析推理结果与标签值的误差有多大；特征数据比如说，C-N键长均匀增加，N上的C绕着N旋转，C-N- O的夹脚发生180+-60度变化等等。不强求，你做的合理就行。
        第二种是推理实验数据，这里参考reference/inference_NMM.py，实验数据保存在bootstrapping/results_240的文件夹中的s0.mat文件，可以考虑将所有的s0_*.mat文件进行合并，然后进行推理。

环境配置：使用了miniconda在当前目录下创建了工作环境。工作时请使用
conda activate ./nmm_ml
我已经安装了requirements.txt中的包. 本地的终端使用的是fish终端。

分子结构：18
NMM ground state equatorial B3LYP/6-311++g**
  C          1.1738643646       -0.2150038459       -1.1426381082    
  C          1.2005823301        0.1916224380        0.3269223126
  H          2.0826129217       -0.2420205126        0.8098032553
  N         -0.0000000094       -0.2957761321        1.0045123511
  H          1.2847492820        1.2935180468        0.3970877198
  C         -1.2005823170        0.1916224591        0.3269223282
  C          0.0000000090        0.0222585972        2.4240707563
  H         -0.0000000090        1.1108999408        2.6209289087
  H          0.8850168643       -0.4095026479        2.8990682950
  H         -0.8850168234       -0.4095026717        2.8990683099
  C         -1.1738643499       -0.2150038773       -1.1426380616
  H         -1.2219012192       -1.3110365455       -1.2237002046
  O         -0.0000000389        0.2643605276       -1.7911323609
  H         -2.0209567441        0.2165532143       -1.6793050474
  H          1.2219012054       -1.3110365132       -1.2237002277
  H          2.0209569312        0.2165531532       -1.6793049808
  H         -1.2847492764        1.2935180740        0.3970877448
  H         -2.0826128243       -0.2420205493        0.8098033280
这里的分子结构是这样的：C1和H15，H16（即第一行，第15行16行）是平面内的一个甲基，平面内另外三个甲基按照逆时针分别是C2H3H5，C6H17H18,C11H11H12,然后氧原子是O13，氮原子是N4，氮上面连的甲基是C7H8H9H10。




4> 使用指南（请保留并按需补充）

一、目录结构（已按本手册在根目录生成）
- manual.md（本文件）
- config.py（核心可改参数集中处）
- requirements.txt（依赖版本）
- params/（保存归一化参数、模型权重、推理输出等）
- file_training_data/（保存训练/验证 h5 数据）
- functions.py（几何/信号/噪声/归一化/数据集/网络）
- train_gen_NMM.py（生成训练数据）
- train.py（训练网络）
- inference.py（推理：合成数据与实验数据）

二、你最常改的参数都在 config.py（建议只改这里）
- S 长度与 s 轴：S_LEN / S_GRID / S_CUTOFF_MIN / S_CUTOFF_MAX
- 原子映射：NMM_ATOM_ORDER / NMM_ATOM_INDEX / LABEL_ATOMS
- 基态坐标：NMM_BASE_COORDS_ANG（请替换为真实 NMM 坐标）
- H 锁死偏移：H_LOCKED_OFFSETS_ANG（请替换为真实偏移；决定 H 参与干涉项）
- 训练数据自由度范围：GEN_R_* / GEN_C5_OOP_DEG_RANGE
- 训练集筛选条件：FILTERS（例如 trilateration 的最小高度阈值、最大面外角）
- 噪声开关与量级：NOISE_*（可叠加高斯/漂移/尖峰）

三、安装环境（示例）
- 建议用 conda 创建环境（python>=3.11），然后在根目录执行：
  pip install -r requirements.txt

四、生成训练数据（先小规模跑通，再扩大到 100 万）
- 小规模（用于验证流程）：
  python train_gen_NMM.py --train_n 20000 --val_n 2000
- 大规模（正式训练，可按你机器时间调整）：
  python train_gen_NMM.py --train_n 1000000 --val_n 20000

生成结果：
- file_training_data/train_NMM.h5
- file_training_data/val_NMM.h5
- params/normalization.json

五、训练模型
- 例子：
  python train.py --epochs 30 --batch_size 128 --lr 2e-4

训练输出：
- params/models/nmm_regressor.pt（包含模型权重与归一化文件路径）

六、推理（两种模式）
1) 合成数据 sanity-check（默认 10000 条）：
  python inference.py --mode synth --synth_n 10000
输出：params/inference_out.npz（含 y_hat / y_true / mae 等）

2) 推理实验数据（自动合并 bootstrapping/results_240 下 s0*.mat）：
  python inference.py --mode exp --exp_dir bootstrapping/results_240 --mat_key s0
输出：params/inference_out.npz（含 y_hat）

七、注意事项（你需要立刻确认/修改的关键点）
1) 当前 config.py 里的 NMM_BASE_COORDS_ANG 与 H_LOCKED_OFFSETS_ANG 是“可运行占位示例”，你必须替换为真实 NMM 结构与 H 的相对偏移，否则信号的物理意义会偏离真实体系。
2) 如果实验数据的 s 轴不是 config.S_GRID（例如你 Matlab 径向积分输出了真实 s 数组），需要把 config.S_GRID 替换为真实 s（或在 inference.py 中对实验数据插值到训练 s 轴）。
3) 若你希望加入“训练集限制条件”（例如键长上下限更严格、某些角度范围更窄、剔除异常构型），优先在 config.py 的 GEN_R_* / FILTERS 里改；如需更复杂筛选，可在 train_gen_NMM.py 的 try/except 处添加规则。

八、采样模块说明（新的几何约束采样 - 2026/3/12）

**1. 采样策略概览**
- 本项目使用**几何约束采样法**而非传统多距离参数化。
- **固定原子**（6个）：O7、C1、C2、C4、C6（及其锁死 H）
- **移动原子**（2个）：N3（圆柱约束）、C5（球约束）
- **核心思想**：N3 始终在距离 O7 为 3.5Å、离 NO 轴向平面 2.0Å 的圆柱面上；C5 始终在距离 N3 为 5.0Å 的球面上。然后检验键长约束：r(N-O) < r(C5-O)（如不满足则重新采样）。

**2. 关键配置参数（config.py, lines 161-172）**
```
GEN_N_CYLINDER_OFF_PLANE_ANG = 2.0      # N3 离 NO 轴向平面的距离（Å）【调试】
GEN_N_CYLINDER_RADIUS_ANG = 3.5         # N3 圆柱采样半径（Å）【调试】
GEN_C5_SPHERE_RADIUS_ANG = 5.0          # C5 球采样半径（Å）【调试】
GEN_CHECK_N_O_SHORTER_THAN_C5_O = True  # 是否检查键长约束
```
注：上面三个参数标记为【调试】，说明这些是经验值，可根据实际化学需求调整。

**3. 采样方法（简要说明）**
- **N3 采样**：在圆柱面上均匀采样，使用 $r = R \sqrt{u}$ 实现面积均匀。
- **C5 采样**：在球面上均匀采样，使用 $r = R u^{1/3}$ 实现体积均匀。
- **键长约束**：采样后检查 r(N-O) < r(C5-O)，不满足则异常，在顶层循环中重新采样。
- **约束通过率**：当前实验数据约 80% 通过，20% 被键长约束滤除（正常行为）。

**4. 顶层采样循环**
- **入口**：`train_gen_NMM.py` 中的 `generate_dataset`。
- **流程**：
  ```
  while len(x_list) < n_samples and tries < max_tries:
    dof = fn.sample_dof(rng)                              # 采样圆柱+球
    try:
      backbone = fn.generate_backbone_coords_from_dof(dof)  # 几何+键长检验
      coords_all = fn.build_full_coords_with_locked_H(backbone)
      signal = fn.compute_1d_scattering_signal(coords_all)
      signal = fn.add_noise(signal, rng)
      label = fn.label_from_backbone(backbone)
      x_list.append(signal)
      y_list.append(label)
    except Exception:
      continue  # 采样失败则重试
  ```
- 当最终 `len(x_list) < n_samples`，会抛出 `RuntimeError`。

**5. 常见问题排查**
- **问**：出现 "only collected 0/200"？
  - **答**：说明所有采样都因键长约束被滤除。可以尝试：
    - 增大 `GEN_C5_SPHERE_RADIUS_ANG`（让 C5 可以离得更远）；
    - 减小 `GEN_N_CYLINDER_RADIUS_ANG`（让 N3 离得近一些）；
    - 调整 `GEN_N_CYLINDER_OFF_PLANE_ANG`；
    - 临时增大 `max_tries_factor` 以便观察筛除比例。

- **问**：想改采样范围怎么办？
  - **答**：所有采样参数都在 config.py 的 lines 161-172，直接改那些【调试】标记的参数即可。

**6. 测试采样实现**
- 可以运行以下命令快速验证采样是否工作正常：
  ```bash
  python test_sampling.py
  ```
  （如果你已经执行过 `train_gen_NMM.py`，该脚本会自动存在。否则可跳过。）
  输出会显示采样成功率、键长约束通过率等信息。
