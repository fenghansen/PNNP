# PNNP (Evaluation Only)
IEEE TPAMI 2026: [Learning Physics-Informed Noise Models from Dark Frames for Low-Light Raw Image Denoising](https://ieeexplore.ieee.org/document/11342300)  

Arxiv (Old Name): [Physics-guided Noise Neural Proxy for Practical Low-light Raw Image Denoising](https://arxiv.org/abs/2310.09126)
## Code Release
> 因忙于毕业论文与项目，近半年分身乏术没有整理，预计7月10日前整理完毕。
没想到中稿花了两年，期间核心技术已应用，不方便完全公开，因此届时预计释放PNNP在SonyA7S2和IMX686上的权重，并释放有PNNP权重时的去噪训练代码。
PNNP本身网络结构极其简单，您可以让gpt或者claude按论文描述复现一下，参数都给了，很容易复现，代码实现效率大概率比我当年还高。

> Due to competing demands from my dissertation and other projects, I have been unable to spare the time to organize the code over the past six months. I expect to complete this by July 10.
As the core technology has already been deployed in commercial applications, the full implementation cannot be made publicly available. Instead, I will release the PNNP weights for the Sony A7S2 and IMX686 sensors, along with the denoising training code that incorporates these PNNP weights.
The PNNP network architecture itself is extremely straightforward. You can easily reproduce it by following the architectural description and parameters provided in the paper—GPT or Claude should handle this effortlessly, and their implementations will likely be more efficient than my original version from years ago.

## Introduction
Currently, this project is only used to evaluate the performance of denoising models trained based on PNNP.  
We also provide a evaluation service for comparative methods, allowing everyone to verify various comparison methods under a unified low-light denoising dataset codebase.   
Currently, supported comparison include:
1. Paired Data (baseline)
3. P-G
4. ELD
5. SFRN
6. NoiseFlow
7. PMN (MM/TPAMI)

We will provide support for additional comparative methods in the future.

## 📋 Prerequisites
* Python >=3.6, PyTorch >= 1.6
* Requirements: opencv-python, rawpy, exifread, h5py, scipy
* Platforms: Ubuntu 16.04, cuda-10.1
* Our method can run on the CPU, but we recommend you run it on the GPU

Please download the datasets first, which are necessary for evaluation (or training).   
ELD ([official project](https://github.com/Vandermode/ELD)): [download (11.46 GB)](https://drive.google.com/file/d/13Ge6-FY9RMPrvGiPvw7O4KS3LNfUXqEX/view?usp=sharing)  
SID ([official project](https://github.com/cchen156/Learning-to-See-in-the-Dark)):  [download (25 GB)](https://storage.googleapis.com/isl-datasets/SID/Sony.zip)  
LRID ([official project](https://fenghansen.github.io/publication/PMN/)):  [download (523 GB)](https://pan.baidu.com/s/1fXlb-Q_ofHOtVOufe5cwDg?pwd=vmcl), including LRID_raw (523 GB, all data), LRID (185.1 GB, for training), results (19.92 GB, PMN visual results) and metrics (59KB, pkl files). **Just download LRID (185.1 GB) is ok.**

***Before the manuscript is accepted, we only provide the weights and results of PNNP for denoising***. You can download them at [[Baidu Netdisk]](https://pan.baidu.com/s/1WMv2x7yqg0kMTBCddqkCLQ?pwd=vmcl).  
`checkpoints` should be downloaded into this project. The arrangement of `resources` can be found in ReadMe.txt under the folder. `samples` contains all the denoising results of PNNP.  
If you choose to save them in a different directory, please remember to update the path location within the respective yaml files (`runfiles/$camera_type$/$method$.yml`).  

## 🎬 Quick Start
1. use `get_dataset_infos.py` to generate dataset infos (please modify `--root_dir`)
```bash 
# Evaluate
python3 get_dataset_infos.py --dstname ELD --root_dir /data/ELD --mode SonyA7S2
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode evaltest
python3 get_dataset_infos.py --dstname LRID --root_dir /data/LRID
# Train
python3 get_dataset_infos.py --dstname SID --root_dir /data/SID/Sony --mode train
# python3 get_dataset_infos.py --dstname LRID --root_dir /data/LRID
```
2. evaluation  

Obviously, you can replace PNNP with other method names. We now provide the weights of P-G, ELD, SFRN, NoiseFlow, LRD, and PMN.
```bash 
# ELD & SID
python3 trainer_SID.py -f runfiles/SonyA7S2/PNNP.yml --mode evaltest
# ELD only
python3 trainer_SID.py -f runfiles/SonyA7S2/PNNP.yml --mode eval
# SID only
python3 trainer_SID.py -f runfiles/SonyA7S2/PNNP.yml --mode test
# LRID
python3 trainer_LRID.py -f runfiles/IMX686/PNNP.yml --mode evaltest
```
If you don't want to save pictures, please add ```--save_plot False```. This option will save your time and space.

3. training (not provided yet)
```bash 
# SID (SonyA7S2)
python3 trainer_PNNP_SID.py -f runfiles/SonyA7S2/Ours.yml --mode train
# LRID (IMX686)
python3 trainer_PNNP_LRID.py -f runfiles/IMX686/Ours.yml --mode train
```

## 🏷️ Citation
Please cite our paper if you find our code helpful in your research or work.
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

## 📧 Contact
If you would like to get in-depth help from me, please feel free to contact me (fenghansen@bit.edu.cn / hansen97@outlook.com) with a brief self-introduction (including your name, affiliation, and position).
