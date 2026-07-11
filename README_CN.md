# PNNP
[English](README.md) | [中文](README_CN.md)

IEEE TPAMI 2026：[Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising](https://ieeexplore.ieee.org/document/11342300)  

Arxiv（旧标题）：[Physics-guided Noise Neural Proxy for Practical Low-light Raw Image Denoising](https://arxiv.org/abs/2310.09126)

## 代码发布
本仓库提供评测代码、Sony A7S2 和 IMX686 的 PNNP 权重，以及使用这些权重合成噪声的去噪训练代码，不包含 PNNP 本体的训练代码。

## 简介
RAW 传感器噪声模型可以写为

$$N = K N_p + N_{indep},$$

其中，信号相关的光子散粒噪声 $N_p$ 可以用泊松分布可靠建模，而信号无关噪声 $N_{indep}$ 更难描述。PNNP 不依赖成对的干净/含噪图像，而是从暗帧中学习这部分噪声。

PNNP 首先将暗帧噪声解耦为帧级、条带级和像素级分量。已知的帧级和条带级分量使用基于物理的标定模型，剩余的 i.i.d. 像素级噪声则由轻量的物理感知神经代理建模。该代理使用空间独立的 $1\times1$ 卷积以及 ISO 相关/ISO 无关双分支，并通过可微分的分位数损失和 CDF 损失训练。发布的权重会与散粒噪声及其他经过标定的物理分量共同使用，为下游 RAW 去噪网络合成训练数据。

目前支持的对比方法包括：

1. Paired Data（基线）
2. P-G
3. ELD
4. SFRN
5. NoiseFlow
6. PMN（MM/TPAMI）

后续将继续支持更多对比方法。

## 📋 环境要求
* Python >= 3.6，PyTorch >= 1.6
* 依赖：opencv-python、rawpy、exifread、h5py、scipy
* 平台：Ubuntu 16.04，CUDA 10.1
* 本方法可以在 CPU 上运行，但建议使用 GPU

请先下载评测或训练所需的数据集。

ELD（[官方项目](https://github.com/Vandermode/ELD)）：[下载（11.46 GB）](https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=sharing)

SID（[官方项目](https://github.com/cchen156/Learning-to-See-in-the-Dark)）：[下载（25 GB）](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)

LRID（[官方项目](https://fenghansen.github.io/publication/PMN/)）：[下载](https://pan.baidu.com/s/1fXlb-Q_ofHOtVOufe5cwDg?pwd=vmcl)，其中包括 LRID_raw（523 GB，全部数据）、LRID（185.1 GB，用于训练）、results（19.92 GB，PMN 可视化结果）和 metrics（59 KB，pkl 文件）。**训练只需下载 LRID（185.1 GB）。**

最新 checkpoint、相机资源、去噪样例以及两个 rawpy 模板均可从[百度网盘](https://pan.baidu.com/s/1WMv2x7yqg0kMTBCddqkCLQ?pwd=vmcl)下载。请将 `checkpoints` 放在项目根目录，并按照压缩包内的 `ReadMe.txt` 放置相机资源。如果使用其他目录，请修改 `runfiles/$camera_type$/$method$.yml` 中的对应路径。

RAW 可视化需要两个模板文件。请将它们直接放在项目根目录：

```text
PNNP/
├── templet.ARW    # Sony A7S2
└── templet.dng    # IMX686
```

它们仅用于向 rawpy 提供生成 RGB 图像所需的 RAW 元数据；启用图像保存后，训练和评测过程会自动加载对应模板。

## 🎬 快速开始
1. 使用 `get_dataset_infos.py` 生成数据集信息文件（按需修改 `--root_dir`）。

```bash
# 评测
python3 get_dataset_infos.py --dstname ELD --root_dir /data/ELD --mode SonyA7S2
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode evaltest
python3 get_dataset_infos.py --dstname LRID --root_dir /data/LRID

# 训练
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode train
python3 get_dataset_infos.py --dstname LRID --root_dir /data/LRID
```

2. 评测

可以将 PNNP 替换成其他方法名。目前提供 P-G、ELD、SFRN、NoiseFlow、LRD 和 PMN 的权重。

```bash
# ELD 和 SID
python3 trainer_SID.py -f runfiles/SonyA7S2/PNNP.yml --mode evaltest
# 仅 ELD
python3 trainer_SID.py -f runfiles/SonyA7S2/PNNP.yml --mode eval
# 仅 SID
python3 trainer_SID.py -f runfiles/SonyA7S2/PNNP.yml --mode test
# LRID
python3 trainer_LRID.py -f runfiles/IMX686/PNNP.yml --mode evaltest
```

如果不需要保存图像，请添加 `--save_plot False`，以节省时间和磁盘空间。

3. 训练

论文设置对每个相机采用连续两个阶段。请先运行 stage 1，再运行 stage 2；stage 2 使用相同的 `model_name` 从第 600 个 epoch 续训至第 1000 个 epoch。

```bash
# Sony A7S2 / SID
bash scripts/train_paper_sony.sh

# IMX686 / LRID
bash scripts/train_paper_imx686.sh
```

如需指定 GPU，可以在运行脚本前设置，例如：

```bash
bash scripts/train_paper_sony.sh 0
```

等价的入口和配置分别为：`trainer_PNNP_SID.py` 搭配 `runfiles/SonyA7S2/PNNP_paper_stage{1,2}.yml`，以及 `trainer_PNNP_LRID.py` 搭配 `runfiles/IMX686/PNNP_paper_stage{1,2}.yml`。

## 🏷️ 引用
如果本项目对您的研究或工作有所帮助，请引用我们的论文。

```bibtex
@article{feng2026learning,
  author={Feng, Hansen and Wang, Lizhi and Huang, Yiqi and Wang, Yuzhi and Zhu, Lin and Huang, Hua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising},
  year={2026},
  volume={48},
  number={4},
  pages={3952-3969}
}
```

## 📧 联系方式
如需进一步帮助，请发送一段简短的自我介绍（包括姓名、单位和职位）至 fenghansen@bit.edu.cn 或 hansen97@outlook.com。
