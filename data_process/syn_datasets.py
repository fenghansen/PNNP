import pickle
import torch
import numpy as np
import cv2
import os
import h5py
import rawpy
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from .unprocess import mosaic, unprocess, random_gains
from .process import *
from utils import *

class SynBase_Dataset(Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__()
        self.default_args()
        if args is not None:
            for key in args:
                self.args[key] = args[key]
        # self.initialization() # 基类不调用

    def default_args(self):
        self.args = {}
        self.args['crop_per_image'] = 8
        self.args['crop_size'] = 512
        self.args['ori'] = False
        self.args['iso'] = None
        self.args['dgain'] = None
        self.args['params'] = None
        self.args['lock_wb'] = False
        self.args['gpu_preprocess'] = False
        self.args['noise_code'] = 'pgrq'
        self.args['dstname'] = 'SID'
        self.args['camera_type'] = 'SonyA7S2'
        self.args['mode'] = 'train'
        self.args['command'] = ''
        self.args['wp'] = 16383
        self.args['bl'] = 512

    def initialization(self):
        # 获取数据地址
        self.suffix = 'ARW'
        self.dataset_file = f'SID_{self.args["mode"]}.info'
        with open(f"infos/{self.dataset_file}", 'rb') as info_file:
            self.infos = pkl.load(info_file)
            print(f'>> Successfully load "{self.dataset_file}" (Length: {len(self.infos)})')
        self.length = len(self.infos)
        self.get_shape()
        self.cache = []
        if 'cache' in self.args['command'].lower():
            log('loading GT into buffer...')
            for idx in tqdm(range(self.length)):
                self.cache.append(np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W))
        self.darkshading = {}
        self.naive = False if '++' in self.args['command'] else True

    def __len__(self):
        return self.length

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W'] 
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4

    def init_random_crop_point(self, mode='non-overlapped', raw_crop=False):
        self.h_start = []
        self.w_start = []
        self.h_end = []
        self.w_end = []
        # 这里比paired-data多了额外的旋转选项，因为行噪声是有方向的
        self.aug = np.random.randint(8, size=self.args['crop_per_image'])
        h, w = self.h, self.w
        if raw_crop:
            h, w = self.H, self.W
        if mode == 'non-overlapped':
            nh = h // self.args["patch_size"]
            nw = w // self.args["patch_size"]
            h_start = np.random.randint(0, h - nh*self.args["patch_size"] + 1)
            w_start = np.random.randint(0, w - nw*self.args["patch_size"] + 1)
            for i in range(nh):
                for j in range(nw):
                    self.h_start.append(h_start + i * self.args["patch_size"])
                    self.w_start.append(w_start + j * self.args["patch_size"])
                    self.h_end.append(h_start + (i+1) * self.args["patch_size"])
                    self.w_end.append(w_start + (j+1) * self.args["patch_size"])

        else: # random_crop
            for i in range(self.args['crop_per_image']):
                h_start = np.random.randint(0, h - self.args["patch_size"] + 1)
                w_start = np.random.randint(0, w - self.args["patch_size"] + 1)
                self.h_start.append(h_start)
                self.w_start.append(w_start)
                self.h_end.append(h_start + self.args["patch_size"])
                self.w_end.append(w_start + self.args["patch_size"])
    
    def data_aug(self, data, mode=0):
        if mode == 0: return data
        rot = mode % 4
        flip = mode // 4
        data = np.rot90(data, k=rot, axes=(-2, -1))
        if flip:
            data = data[..., ::-1]
        return data

    def eval_crop(self, data, base=64):
        crop_size = self.args["patch_size"]
        # crop setting
        d = base//2
        l = crop_size - base
        nh = self.h // l + 1
        nw = self.w // l + 1
        data = F.pad(data, (d, d, d, d), mode='reflect')
        croped_data = torch.empty((nh, nw, self.c, crop_size, crop_size),
                    dtype=data.dtype, device=data.device)
        # 分块crop主体区域
        for i in range(nh-1):
            for j in range(nw-1):
                croped_data[i][j] = data[..., i*l:i*l+crop_size,j*l:j*l+crop_size]
        # 补边
        for i in range(nh-1):
            j = nw - 1
            croped_data[i][j] = data[..., i*l:i*l+crop_size,-crop_size:]
        for j in range(nw-1):
            i = nh - 1
            croped_data[i][j] = data[..., -crop_size:,j*l:j*l+crop_size]
        # 补角
        croped_data[nh-1][nw-1] = data[..., -crop_size:,-crop_size:]
        # 整合为tensor
        croped_data = croped_data.view(-1, self.c, crop_size, crop_size)
        return croped_data
    
    def eval_merge(self, croped_data, base=64):
        crop_size = self.args["patch_size"]
        data = torch.empty((1, self.c, self.h, self.w), dtype=croped_data.dtype, device=croped_data.device)
        # crop setting
        d = base//2
        l = crop_size - base
        nh = self.h // l + 1
        nw = self.w // l + 1
        croped_data = croped_data.view(nh, nw, self.c, crop_size, crop_size)
        # 分块crop主体区域
        for i in range(nh-1):
            for j in range(nw-1):
                data[..., i*l:i*l+l,j*l:j*l+l] = croped_data[i, j, :, d:-d, d:-d]
        # 补边
        for i in range(nh-1):
            j = nw - 1
            data[..., i*l:i*l+l, -l:] = croped_data[i, j, :, d:-d, d:-d]
        for j in range(nw-1):
            i = nh - 1
            data[..., -l:, j*l:j*l+l] = croped_data[i, j, :, d:-d, d:-d]
        # 补角
        data[..., -l:, -l:] = croped_data[nh-1, nw-1, :, d:-d, d:-d]
        
        return data

    # 因为视频随机crop位置是固定的，所以同一个视频的任意分量都可以调用random_crop函数
    def random_crop(self, img):
        # 本函数用于将numpy随机裁剪成以crop_size为边长的方形crop_per_image等份
        c, h, w = img.shape
        # 创建空numpy做画布, [crops, h, w]
        crops = np.empty((self.args["crop_per_image"], c, self.args["patch_size"], self.args["patch_size"]), dtype=np.float32)
        # 往空tensor的通道上贴patchs
        for i in range(self.args["crop_per_image"]):
            crop = img[:, self.h_start[i]:self.h_end[i], self.w_start[i]:self.w_end[i]]
            crop = self.data_aug(crop, mode=self.aug[i])
            crops[i] = crop

        return crops

    # # Naive darkshading
    # def get_darkshading(self, iso):
    #     if iso not in self.darkshading:
    #         self.darkshading[iso] = np.load(os.path.join(self.args['ds_dir'], f'darkshading-iso-{iso}.npy'))
    #     return self.darkshading[iso]
    def get_darkshading(self, iso, exp=25, naive=True, num=None, remake=False):
        branch = '_highISO' if iso>1600 else '_lowISO'
        if iso not in self.darkshading or remake is True:
            ds_path = os.path.join(self.args['ds_dir'], f'darkshading-iso-{iso}.npy')
            ds_k = np.load(os.path.join(self.args['ds_dir'], f'darkshading{branch}_k.npy'))
            ds_b = np.load(os.path.join(self.args['ds_dir'], f'darkshading{branch}_b.npy'))
            if naive: # naive linear darkshading - BLE(ISO)
                with open(os.path.join(self.args['ds_dir'], f'darkshading_BLE.pkl'), 'rb') as f:
                    self.blc_mean = pkl.load(f)
                BLE = self.blc_mean[iso]
            else: # new linear darkshading - BLE(ISO, t)
                with open(os.path.join(self.args['ds_dir'], f'BLE_t.pkl'),'rb') as f:
                    self.blc_mean = pickle.load(f)
                # BLE bias only here
                # kt = np.poly1d(self.blc_mean[f'kt{branch}'])
                BLE = self.blc_mean[iso]['b'] # + kt(iso) * exp
            # D_{ds} = D_{FPNk} + D_{FPNb} + BLE(ISO, t)
            self.darkshading[iso] = ds_k * iso + ds_b + BLE

        if naive:
            return self.darkshading[iso]
        else:
            kt = np.poly1d(self.blc_mean[f'kt{branch}'])
            BLE = kt(iso) * exp
            return self.darkshading[iso] + BLE

# Unprocess Synthetic Dataset(sRGB->Raw)
class Img_Dataset(SynBase_Dataset):
    def __init__(self, args=None):
        # @ noise_code: g,Guassian->TL; p,Guassian->Possion; r,Row; q,Quantization
        super().__init__(args)
        self.args = args if args is not None else self.default_args()
        self.initialization()
    
    def default_args(self):
        super().default_args()
        print('数据集没打包呢')
        raise NotImplementedError

    def initialization(self):
        super().initialization()

    def get_shape(self):
        self.H, self.W = self.args['H'], self.args['W']
        self.C = 3
        self.h = self.H // 2
        self.w = self.W // 2
        self.c = 4

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['name'] = self.infos[idx]['name']
        hr_imgs = np.array(self.f.get(self.infos[idx]['data']))
        hr_imgs = np.frombuffer(hr_imgs, np.uint16).reshape(self.C,self.H,self.W)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=True)
            hr_crops = self.random_crop(hr_imgs)
        elif self.args["mode"] == 'eval':
            hr_crops = hr_imgs[None,:]
    
        # RAW需要复杂的unproces
        lr_shape = hr_crops.shape
        metadata = [None] * lr_shape[0]
        wbs = [None] * lr_shape[0]
        hr_crops = torch.from_numpy(hr_crops)
        if self.args["lock_wb"]:
            wb = self.infos[idx]['wb']
            red_gain, blue_gain = wb[0], wb[2]
            lock_wb = ([1.], [red_gain], [blue_gain])
        else:
            lock_wb = False

        for i in range(lr_shape[0]):
            hr_crops[i], metadata[i] = unprocess(hr_crops[i], lock_wb=lock_wb, use_gpu=self.args['gpu_preprocess'])
            wbs[i] = np.array([metadata['red_gain'].numpy(), 1., metadata['blue_gain'].numpy()])
        hr_crops = mosaic(hr_crops).numpy()
        # [crops,h,w,c] -> [crops,c,h,w]
        hr_crops = hr_crops.transpose(0,3,1,2)
        lr_crops = hr_crops.copy()
        data['ccm'] = metadata['cam2rgb'].numpy()
        data['wb'] = np.array(wbs)

        # 人工加噪声
        data['ratio'] = np.ones(lr_shape[0], dtype=np.float32)
        if self.args['gpu_preprocess'] is False:
            for i in range(lr_shape[0]):
                if self.args['params'] is None:
                    # 这里可以兼容训练单ISO/dgain的模型
                    param_iso = self.args['iso']
                    param_dgain = self.args['dgain']
                    noise_param = sample_params(camera_type=self.args['camera_type'], ratio=param_dgain, iso=param_iso)
                else:
                    noise_param = self.args['params']
                data['ratio'][i] = noise_param['ratio']
                lr_crops[i] = generate_noisy_obs(lr_crops[i], param=noise_param, noise_code=self.args['noise_code'], ori=self.args['ori'])
        
        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

# Standard Synthetic Dataset (Raw2Raw)
class Raw_Dataset(SynBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']
        hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            hr_crops = self.random_crop(hr_imgs)
        elif self.args["mode"] == 'eval':
            hr_crops = hr_imgs[None,:]
        lr_shape = hr_crops.shape
        
        if self.args["lock_wb"] is False and np.random.randint(2):
            rgb_gain, red_gain, blue_gain = random_gains()
            red_gain = data['wb'][0] / red_gain.numpy()
            blue_gain = data['wb'][2] / blue_gain.numpy()
            hr_crops *= rgb_gain.numpy()
            hr_crops[:,0] = hr_crops[:,0] * red_gain
            hr_crops[:,2] = hr_crops[:,2] * blue_gain
            # data['rgb_gain'] = np.ones(lr_shape[0], dtype=np.float32) * rgb_gain.numpy()

        lr_crops = hr_crops.copy()
        # 人工加噪声
        data['ratio'] = np.ones(lr_shape[0], dtype=np.float32)
        if self.args['gpu_preprocess'] is False:
            for i in range(lr_shape[0]):
                if self.args['params'] is None:
                    # # sample_params_max训练单ISO/dgain的模型
                    # param_iso = self.args['iso']
                    # param_dgain = self.args['dgain']
                    noise_param = sample_params(camera_type=self.args['camera_type'])
                else:
                    noise_param = self.args['params']
                if 'GTdn' in self.args['command']:
                    noise_param['ratio'] = np.maximum(np.random.uniform(-3,4), 1)
                data['ratio'][i] = noise_param['ratio']
                lr_crops[i] = generate_noisy_obs(lr_crops[i], param=noise_param, noise_code=self.args['noise_code'], ori=self.args['ori'])
        
        if self.args['clip']:
            lb = -np.inf if self.args['clip']==HALF_CLIP else 0
            lr_crops = lr_crops.clip(lb,1)
            hr_crops = hr_crops.clip(0, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

# Learning-based Synthetic Dataset (Raw2Raw)
class NF_Syn_Dataset(SynBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']
        data['ISO'] = self.infos[idx]['ISO']
        if 'cache' in self.args['command'].lower():
            hr_raw = self.cache[idx]
        else:
            hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            hr_crops = self.random_crop(hr_imgs)
        elif self.args["mode"] == 'eval':
            hr_crops = hr_imgs[None,:]
        lr_shape = hr_crops.shape
        
        if self.args["lock_wb"] is False and np.random.randint(2):
            rgb_gain, red_gain, blue_gain = random_gains()
            red_gain = data['wb'][0] / red_gain.numpy()
            blue_gain = data['wb'][2] / blue_gain.numpy()
            hr_crops *= rgb_gain.numpy()
            hr_crops[:,0] = hr_crops[:,0] * red_gain
            hr_crops[:,2] = hr_crops[:,2] * blue_gain
            # data['rgb_gain'] = np.ones(lr_shape[0], dtype=np.float32) * rgb_gain.numpy()

        lr_crops = hr_crops.copy()
        # 人工加噪声
        data['ratio'] = np.ones(lr_shape[0], dtype=np.float32)
        if self.args['gpu_preprocess'] is False:
            raise NotImplementedError
        
        if self.args['clip']:
            lb = -np.inf if self.args['clip']==HALF_CLIP else 0
            lr_crops = lr_crops.clip(lb,1)
            hr_crops = hr_crops.clip(0, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

# Same as NF_Syn_Dataset
class Proxy_Dataset(SynBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']
        if 'cache' in self.args['command'].lower():
            hr_raw = self.cache[idx]
        else:
            hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args["mode"] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            hr_crops = self.random_crop(hr_imgs)
        elif self.args["mode"] == 'eval':
            hr_crops = hr_imgs[None,:]
        lr_shape = hr_crops.shape
        
        if self.args["lock_wb"] is False and np.random.randint(2):
            rgb_gain, red_gain, blue_gain = random_gains()
            red_gain = data['wb'][0] / red_gain.numpy()
            blue_gain = data['wb'][2] / blue_gain.numpy()
            hr_crops *= rgb_gain.numpy()
            hr_crops[:,0] = hr_crops[:,0] * red_gain
            hr_crops[:,2] = hr_crops[:,2] * blue_gain
            # data['rgb_gain'] = np.ones(lr_shape[0], dtype=np.float32) * rgb_gain.numpy()

        lr_crops = hr_crops.copy()
        # 人工加噪声
        data['ratio'] = np.ones(lr_shape[0], dtype=np.float32)
        if self.args['gpu_preprocess'] is False:
            raise NotImplementedError
        
        if self.args['clip']:
            lb = -np.inf if self.args['clip']==HALF_CLIP else 0
            lr_crops = lr_crops.clip(lb,1)
            hr_crops = hr_crops.clip(0, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

# SFRN
class SFRN_Dataset(SynBase_Dataset):
    def __init__(self, args=None):
        super().__init__(args)
        self.initialization()
    
    def default_args(self):
        super().default_args()

    def initialization(self):
        super().initialization()
        # 创建HighBitRecovery类
        self.HBR = HighBitRecovery(camera_type=self.args['camera_type'], noise_code=self.args['noise_code'])
        log('Recording Bias Frames...')
        self.black_dirs = sorted(os.listdir(self.args['bias_dir']), key=lambda x: int(x))
        self.legalISO = [int(dirname) for dirname in self.black_dirs]
        self.black_dirs = [os.path.join(self.args['bias_dir'], dirname) for dirname in self.black_dirs]
        self.blacks = [None] * len(self.legalISO)
        for i in range(len(self.legalISO)):
            self.blacks[i] = [os.path.join(self.black_dirs[i], filename) for filename in os.listdir(self.black_dirs[i])]

        self.darkshading = {}
        self.blc_mean = {}
        for iso in self.legalISO:
            if self.naive:
                if os.path.exists(os.path.join(self.args['ds_dir'], f'darkshading_BLE.pkl')):
                    with open(os.path.join(self.args['ds_dir'], f'darkshading_BLE.pkl'), 'rb') as f:
                        self.blc_mean = pkl.load(f)
                self.get_darkshading(iso)
                self.blc_mean[iso] = raw2bayer(self.darkshading[iso], norm=False, clip=False, 
                                        wp=self.args['wp']-self.args['bl'], bl=0)
                self.blc_mean[iso] = np.mean(self.blc_mean[iso])
            else:
                self.get_darkshading(iso, naive=self.naive)

        if 'darkshading' in self.args['command']:
            # darkshaidng会矫正输入，不能算减去的分布
            self.HBR.get_lut(self.legalISO, blc_mean=None)
        elif 'blc' in self.args['command']:
            # 只使用均值矫正
            self.HBR.get_lut(self.legalISO, self.blc_mean)
        else:
            # 假装不知道有bias这回事
            self.HBR.get_lut(self.legalISO, blc_mean=None)

    def __getitem__(self, idx):
        data = {}
        # 读取数据
        data['wb'] = self.infos[idx]['wb']
        data['ccm'] = self.infos[idx]['ccm']
        data['name'] = self.infos[idx]['name']
        iso_index = np.random.randint(len(self.legalISO))
        data['ISO'] = self.legalISO[iso_index]
        
        hr_raw = np.array(dataload(self.infos[idx]['long'])).reshape(self.H,self.W)
        lr_id = np.random.randint(len(self.blacks[iso_index])) if self.args['mode']=='train' else 0
        if 'lr10' in self.args['command']:
            lr_id = np.random.randint(10)
        lr_raw = rawpy.imread(self.blacks[iso_index][lr_id]).raw_image_visible

        data['ExposureTime'] = self.infos[idx]['ExposureTime'] * 1000
        data['exp'] = data['ExposureTime'] / (100 + random.random()*200)

        if 'darkshading' in self.args['command']:
            lr_raw = lr_raw - self.get_darkshading(iso=data['ISO'], exp=data['exp'], naive=self.naive)

        if 'blc2' in self.args['command']:
            hr_raw = hr_raw - self.blc_mean[data['ISO']]
        lr_imgs = raw2bayer(lr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=False)
        hr_imgs = raw2bayer(hr_raw, wp=self.args['wp'], bl=self.args['bl'], norm=True, clip=True)

        if self.args['mode'] == 'train':
            # 随机裁剪成crop_per_image份
            self.init_random_crop_point(mode=self.args['croptype'], raw_crop=False)
            hr_crops = self.random_crop(hr_imgs)
            lr_crops = self.random_crop(lr_imgs)
            # Raw wb_aug
            if self.args['lock_wb'] is False and np.random.randint(2):
                wb = data["wb"]
                rgb_gain, red_gain, blue_gain = random_gains()
                red_gain = wb[0] / red_gain.numpy()
                blue_gain = wb[2] / blue_gain.numpy()
                hr_crops *= rgb_gain.numpy()
                hr_crops[:,0] = hr_crops[:,0] * red_gain
                hr_crops[:,2] = hr_crops[:,2] * blue_gain

            lr_shape = lr_crops.shape
            data['ratio'] = np.random.uniform(low=100, high=300, size=lr_shape[0])

            # 贴信号无关的read noise! HB矫正！
            if 'preHB' not in self.args['command'] and 'HB' in self.args['command']:
                lr_crops = self.HBR.map(lr_crops, data['ISO'], norm=True)

            # 加信号相关的shot noise
            if not self.args['gpu_preprocess']:
                for i in range(lr_shape[0]):
                    if self.args['params'] is None:
                        # 按ISO生成噪声参数（其实只要泊松）
                        noise_param = {'K': 0.0009546 * data['ISO'] * (1 + np.random.uniform(low=-0.01, high=+0.01)) - 0.00193}
                        noise_param['wp'], noise_param['bl'] = self.args['wp'], self.args['bl']
                        noise_param['ratio'] = np.random.uniform(low=100, high=300)
                    else:
                        noise_param = self.args['params']
                    data['ratio'][i] = noise_param['ratio']
                    lr_crops[i] += generate_noisy_obs(hr_crops[i], param=noise_param, 
                                                    noise_code=self.args['noise_code']+'b', ori=self.args['ori'])

        if self.args['clip']:
            lb = -np.inf if self.args['clip']==HALF_CLIP else 0
            lr_crops = lr_crops.clip(lb,1)
            hr_crops = hr_crops.clip(0, 1)

        data["lr"] = np.ascontiguousarray(lr_crops)
        data["hr"] = np.ascontiguousarray(hr_crops)
        
        return data

if __name__=='__main__':
    pass