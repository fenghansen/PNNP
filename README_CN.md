# PNNP
[English](README.md) | [中文](README_CN.md)

**[论文 (IEEE TPAMI)](https://ieeexplore.ieee.org/document/11342300)** | **[arXiv](https://arxiv.org/abs/2310.09126)** | **[LRID 数据集 (Hugging Face)](https://huggingface.co/datasets/hansen97/LRID)** | **[模型权重与资源 (百度网盘)](https://pan.baidu.com/s/1WMv2x7yqg0kMTBCddqkCLQ?pwd=vmcl)**

IEEE TPAMI 2026：[Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising](https://ieeexplore.ieee.org/document/11342300)  

Arxiv（旧标题）：[Physics-guided Noise Neural Proxy for Practical Low-light Raw Image Denoising](https://arxiv.org/abs/2310.09126)

## 🎉 新闻
* **(2026.04)** 🎉 论文发表于 *IEEE Transactions on Pattern Analysis and Machine Intelligence*（Vol. 48, No. 4, pp. 3952-3969）。
* **(2026.03)** 🗃️ LRID 数据集发布至 Hugging Face：[hansen97/LRID](https://huggingface.co/datasets/hansen97/LRID)（完整版）与 [hansen97/LRID_simplified](https://huggingface.co/datasets/hansen97/LRID_simplified)（用于快速验证的精简版）。
* **(2023.10)** 📰 预印本发布于 arXiv。

## ✨ 亮点
* **从暗帧学习噪声模型。** 我们提出从暗帧（而非成对真实数据）中学习传感器噪声模型，打破了基于学习的噪声建模对数据的依赖。
* **物理引导的噪声解耦（PND）。** 将暗帧噪声解耦为帧级、条带级和像素级分量，已知的帧级与条带级分量由基于物理的标定处理，降低了噪声建模的复杂度。
* **物理感知的代理模型（PPM）。** 轻量神经代理使用空间独立的 $1\times1$ 卷积以及 ISO 相关/ISO 无关双分支，对剩余的 i.i.d. 像素级噪声建模，并以物理先验约束，提升噪声建模的准确性。
* **可微分的分布损失（DDL）。** 可微分的分位数损失与 CDF 损失为噪声分布提供了显式、可靠的监督，提升噪声建模的精度。

## 📦 代码发布
本仓库提供评测代码、Sony A7S2 和 IMX686 的 PNNP 权重，以及使用这些权重合成噪声的去噪训练代码，不包含 PNNP 本体的训练代码。

## 📖 简介
RAW 传感器噪声模型可以写为

$$N = K N_p + N_{indep},$$

其中，信号相关的光子散粒噪声 $N_p$ 可以用泊松分布可靠建模，而信号无关噪声 $N_{indep}$ 更难描述。PNNP 不依赖成对的干净/含噪图像，而是从暗帧中学习这部分噪声。

![RAW 传感器成像流水线中的噪声形成过程](images/github/NFM.jpg)

PNNP 首先将暗帧噪声解耦为帧级、条带级和像素级分量。已知的帧级和条带级分量使用基于物理的标定模型，剩余的 i.i.d. 像素级噪声则由轻量的物理感知神经代理建模。该代理使用空间独立的 $1\times1$ 卷积以及 ISO 相关/ISO 无关双分支，并通过可微分的分位数损失和 CDF 损失训练。发布的权重会与散粒噪声及其他经过标定的物理分量共同使用，为下游 RAW 去噪网络合成训练数据。

![PNNP 框架总览](images/github/framework.jpg)

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

## 🗃️ 数据集与资源
请先下载评测或训练所需的数据集。

### 数据集
* **ELD**（[官方项目](https://github.com/Vandermode/ELD)）：[下载（11.46 GB）](https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=sharing)
* **SID**（[官方项目](https://github.com/cchen156/Learning-to-See-in-the-Dark)）：[下载（25 GB）](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)
* **LRID**（[官方项目](https://fenghansen.github.io/publication/PMN/)）：我们此前工作 [PMN](https://github.com/megvii-research/PMN/tree/TPAMI)（ACM MM 2022 / IEEE TPAMI 2024）使用 IMX686 手机相机采集的低光 RAW 去噪数据集，本仓库将其用于 IMX686 相机的训练与评测。
  * **百度网盘**：[下载](https://pan.baidu.com/s/1fXlb-Q_ofHOtVOufe5cwDg?pwd=vmcl)，其中包括 LRID_raw（523 GB，全部数据）、LRID（185.1 GB，用于训练）、results（19.92 GB，PMN 可视化结果）和 metrics（59 KB，pkl 文件）。**训练只需下载 LRID（185.1 GB）。**
  * **Hugging Face**：[hansen97/LRID](https://huggingface.co/datasets/hansen97/LRID) 托管了完整的 LRID 数据集（全部原始数据，包括暗帧、参考帧和中间结果）；[hansen97/LRID_simplified](https://huggingface.co/datasets/hansen97/LRID_simplified) 每个场景仅保留一帧含噪图像，同时保留全部真值与暗帧，适用于快速验证而非完整训练。

### 暗帧
PNNP 从暗帧中学习噪声模型，为新相机标定或学习噪声模型时需要暗帧数据：
* **IMX686**：暗帧已包含在 LRID 数据集中（`bias/` 与 `bias-hot/`）。
* **Sony A7S2**：推荐使用 **LLD 数据集**（[官方项目](https://github.com/happycaoyue/LLD)；CVPR 2023，*Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography*）中的暗帧，该数据集提供了由 Sony A7S2 在广泛 ISO 范围内拍摄的暗（偏置）帧：[下载（百度网盘）](https://pan.baidu.com/s/1eLKzjOSDCR4NNLcvbyenEQ?pwd=WAXY)。如果使用这些暗帧，请同时引用 LLD 论文（见[引用](#引用)）。

### 模型权重、相机资源与 rawpy 模板
最新 checkpoint、相机资源、去噪样例以及两个 rawpy 模板均可从[百度网盘](https://pan.baidu.com/s/1WMv2x7yqg0kMTBCddqkCLQ?pwd=vmcl)下载。请将 `checkpoints` 放在项目根目录，并按照压缩包内的 `ReadMe.txt` 放置相机资源。如果使用其他目录，请修改 `runfiles/$camera_type$/$method$.yml` 中的对应路径。

RAW 可视化需要两个模板文件。请将它们直接放在项目根目录：

```text
PNNP/
├── templet.ARW    # Sony A7S2
└── templet.dng    # IMX686
```

它们仅用于向 rawpy 提供生成 RGB 图像所需的 RAW 元数据；启用图像保存后，训练和评测过程会自动加载对应模板。

## 🎬 快速开始
### 1. 生成数据集信息
使用 `get_dataset_infos.py` 生成数据集信息文件（按需修改 `--root_dir`）。

```bash
# 评测
python3 get_dataset_infos.py --dstname ELD --root_dir /data/ELD --mode SonyA7S2
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode evaltest
python3 get_dataset_infos.py --dstname LRID --root_dir /data/LRID

# 训练
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode train
python3 get_dataset_infos.py --dstname LRID --root_dir /data/LRID
```

### 2. 评测
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

### 3. 训练
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

## 📄 结果
SID 数据集（Sony A7S2）上的可视化对比，各方法下方的数值为 PSNR/SSIM。

![SID 数据集上的可视化对比](images/github/results_SID.jpg)

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

如果使用了 LRID 数据集或 LLD 暗帧，请同时引用相应的论文：

```bibtex
@inproceedings{feng2022learnability,
  author={Feng, Hansen and Wang, Lizhi and Wang, Yuzhi and Huang, Hua},
  title={Learnability Enhancement for Low-Light Raw Denoising: Where Paired Real Data Meets Noise Modeling},
  booktitle={Proceedings of the 30th ACM International Conference on Multimedia},
  year={2022},
  pages={1436--1444},
  numpages={9},
  location={Lisboa, Portugal},
  series={MM '22}
}

@article{feng2023learnability,
  author={Feng, Hansen and Wang, Lizhi and Wang, Yuzhi and Fan, Haoqiang and Huang, Hua},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Learnability Enhancement for Low-Light Raw Image Denoising: A Data Perspective},
  year={2024},
  volume={46},
  number={1},
  pages={370-387},
  doi={10.1109/TPAMI.2023.3301502}
}

@inproceedings{cao2023physics,
  author={Cao, Yue and Liu, Ming and Liu, Shuai and Wang, Xiaotao and Lei, Lei and Zuo, Wangmeng},
  title={Physics-Guided ISO-Dependent Sensor Noise Modeling for Extreme Low-Light Photography},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  month={June},
  year={2023},
  pages={5744-5753}
}
```

## 📧 联系方式
如需进一步帮助，请发送一段简短的自我介绍（包括姓名、单位和职位）至 fenghansen@bit.edu.cn 或 hansen97@outlook.com。
