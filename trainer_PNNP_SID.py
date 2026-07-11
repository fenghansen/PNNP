import os
import time
from torch.optim import Adam, lr_scheduler
from data_process import *
from utils import *
from archs import *
from losses import *
from base_trainer import *

class PNNP_SID_Trainer(Base_Trainer):
    def __init__(self):
        parser = PNNP_SID_Parser()
        self.parser = parser.parse()
        self.initialization()
        # model
        self.arch_proxy = self.args['arch_proxy'] if 'arch_proxy' in self.args else None
        # self.arch_proxy['noise_code'] = self.dst['noise_code'] # 同步噪声参数选项
        # 初始化网络
        self.net = globals()[self.arch['name']](self.arch)
        if self.arch_proxy['name'] == 'NoiseFlow':
            self.proxy = self.args['arch_proxy']
            self.proxy_net = globals()[self.proxy['name']](self.proxy)
            model_path = os.path.join(f'{self.fast_ckpt}/pw_iso_noiseflow.pth')
            model = torch.load(model_path, map_location=self.device)
            self.proxy_net = load_weights(self.proxy_net, model, by_name=True).to(self.device)
            self.proxy_net.eval()
            self.legalISO = torch.tensor([50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600,
                                   2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600]).to(self.device)
        else:
            self.load_proxynet()
        # load model weights
        if self.hyper['last_epoch']:    # 不是初始化
            try:
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
                if not os.path.exists(model_path):
                    model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
                model = torch.load(model_path, map_location=self.device)
                self.net = load_weights(self.net, model, by_name=True)
            except:
                log('No checkpoint file!!!')
        else:
            log(f'Initializing {self.arch["name"]}...')
            # initialize_weights(self.net)

        self.optimizer = Adam(self.net.parameters(), lr=self.hyper['learning_rate'])
        
        self.infos = None
        if self.mode=='train':
            self.dst_train = globals()[self.args['dst_train']['dataset']](self.args['dst_train'])
            self.dataloader_train = DataLoader(self.dst_train, batch_size=self.hyper['batch_size'], worker_init_fn=worker_init_fn,
                                    shuffle=True, num_workers=self.args['num_workers'], pin_memory=True)
            self.change_eval_dst('eval')
            self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=True, 
                                    num_workers=self.args['num_workers'], pin_memory=False)

        # Choose Learning Rate
        self.lr_lambda = self.get_lr_lambda_func()
        self.scheduler = LambdaScheduler(self.optimizer, self.lr_lambda)

        self.net = self.net.to(self.device)
        self.loss = Unet_Loss()
        self.corrector = IlluminanceCorrect()
        
        # Print Model Log
        self.print_model_log()
    
    def change_eval_dst(self, mode='eval'):
        self.dst = self.args[f'dst_{mode}']
        self.dstname = self.dst['dstname']
        self.dst_eval = globals()[self.dst['dataset']](self.dst)
        self.dataloader_eval = DataLoader(self.dst_eval, batch_size=1, shuffle=False, 
                                    num_workers=self.args['num_workers'], pin_memory=False)
        self.cache_dir = f'/data/cache/{self.dstname}'

    def train(self):
        self.scheduler.step()
        lr = self.scheduler.get_last_lr()[0]
        for epoch in range(self.hyper['last_epoch']+1, self.hyper['stop_epoch']+1):
            # log init
            self.net.train()
            self.train_psnr.reset()
            runtime = {'preprocess':0, 'dataloader':0, 'net':0, 'bp':0, 'metric':0, 'total':1e-9}
            time_points = [0] * 10
            time_points[0] = time.time()   
            with tqdm(total=len(self.dataloader_train)) as t:                
                for k, data in enumerate(self.dataloader_train):
                    runtime['dataloader'] += timestamp(time_points, 1)
                    # Preprocess
                    imgs_lr, imgs_hr, ratio = self.preprocess(data, mode='train', preprocess=True)
                    runtime['preprocess'] += timestamp(time_points, 2)
                    
                    # 训练
                    self.optimizer.zero_grad()
                    pred = self.net(imgs_lr)
                    runtime['net'] += timestamp(time_points, 3)
                    # 极暗，乘上去
                    if self.dst['ori'] is True:
                        pred = pred * ratio
                    loss = self.loss(pred.clamp(0,1), imgs_hr)
                    loss.backward()
                    self.optimizer.step()
                    runtime['bp'] += timestamp(time_points, 4)

                    # 更新tqdm的参数
                    with torch.no_grad():
                        if self.arch['use_dpsv']: 
                            pred = pred[0]
                        if 'rgb_gain' in data:
                            data['rgb_gain'] = data['rgb_gain'].view_as(ratio)
                            pred = pred / data['rgb_gain']
                            imgs_hr = imgs_hr / data['rgb_gain']
                        pred = torch.clamp(pred, 0, 1)
                        imgs_hr = torch.clamp(imgs_hr, 0, 1)
                        psnr = PSNR_Loss(pred, imgs_hr)
                        self.train_psnr.update(psnr.item())
                    
                    runtime['total'] = runtime['preprocess']+runtime['dataloader']+runtime['net']+runtime['bp']
                    t.set_description(f'Epoch {epoch}')
                    t.set_postfix({'lr':f"{lr:.2e}", 'PSNR':f"{self.train_psnr.avg:.2f}",
                                    'loader':f"{100*runtime['dataloader']/runtime['total']:.1f}%",
                                    'process':f"{100*runtime['preprocess']/runtime['total']:.1f}%",
                                    'net':f"{100*runtime['net']/runtime['total']:.1f}%",
                                    'bp':f"{100*runtime['bp']/runtime['total']:.1f}%",})
                    t.update(1)
                    time_points[0] = time.time()

            # 更新学习率
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]

            # 存储模型
            if epoch % self.hyper['save_freq'] == 0:
                model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                epoch_id = epoch // self.hyper['plot_freq'] * self.hyper['plot_freq']
                save_path = os.path.join(self.model_dir, '%s_e%04d.pth'% (self.model_name, epoch_id))
                torch.save(model_dict, save_path)
            
            # 输出过程量，随时看
            savefile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.jpg')
            logfile = os.path.join(self.sample_dir, f'{self.model_name}_train_psnr.pkl')
            self.train_psnr.plot_history(savefile=savefile, logfile=logfile)
            # if epoch % self.hyper['plot_freq'] == 0:
            wb = data['wb'][0].numpy()
            if self.dst['ori'] is True:
                imgs_lr = imgs_lr * ratio
                pred = pred
                imgs_hr = imgs_hr# * ratio
            
            if self.save_plot:
                inputs = imgs_lr[0].detach().cpu().numpy().clip(0,1)
                output = pred[0].detach().cpu().numpy()
                target = imgs_hr[0].detach().cpu().numpy()
                temp_img = np.concatenate((inputs, output, target),axis=2)[:3]
                temp_img[0] = temp_img[0] * wb[0]
                temp_img[2] = temp_img[2] * wb[2]
                filename = os.path.join(self.sample_dir, 'temp', f'temp_{epoch//10*10:04d}.png')
                temp_img = temp_img.transpose(1,2,0)[:,:,::-1] ** (1/2.2)
                cv2.imwrite(filename, np.uint8(temp_img*255))

            # fast eval
            if epoch % self.hyper['plot_freq'] == 0:
                log(f"learning_rate: {lr:.3e}")
                self.dst_eval.fast_eval(on=True)
                self.eval(epoch=epoch)
                self.dst_eval.fast_eval(on=False)
                model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
                torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_last_model.pth')
            
            # reload best model each period
            num_of_epochs = self.hyper['stop_epoch'] - self.hyper['last_epoch']
            T = self.hyper['T'] if 'T' in self.hyper else 1 
            period = num_of_epochs//T
            if (self.hyper['last_epoch']+epoch) % period == 0:
                model_path = os.path.join(f'{self.fast_ckpt}/{self.model_name}_best_model.pth')
                if os.path.exists(model_path):
                    model = torch.load(model_path, map_location=self.device)
                    self.net = load_weights(self.net, model, by_name=True)
                    log(f'Successfully reload best model (Eval PSNR:{self.best_psnr})',
                        log=f'./logs/log_{self.model_name}.log')

    def eval(self, epoch=-1):
        self.net.eval()
        self.eval_psnr.reset()
        self.eval_ssim.reset()
        self.eval_psnr_lr.reset()
        self.eval_psnr_dn.reset()
        self.eval_ssim_lr.reset()
        self.eval_ssim_dn.reset()
        # record every metric
        metrics = {}
        metrics_path = f'./metrics/{self.model_name}_metrics.pkl'
        if os.path.exists(metrics_path):
            with open(metrics_path, 'rb') as f:
                metrics = pkl.load(f)
        # multiprocess
        if epoch > 0:
            pool = []
        else:
            pool = ProcessPoolExecutor(max_workers=max(4, self.args['num_workers']))
        task_list = []
        save_plot = self.save_plot
        with tqdm(total=len(self.dataloader_eval)) as t:
            for k, data in enumerate(self.dataloader_eval):
                # 由于crops的存在，Dataloader会把数据变成5维，需要view回4维
                imgs_lr, imgs_hr, ratio = self.preprocess(data, mode='eval', preprocess=False)
                wb = data['wb'][0].numpy()
                ccm = data['ccm'][0].numpy()
                name = data['name'][0]
                ISO = data['ISO'].item()
                exp = data['ExposureTime'].item()
                # print(ISO)

                with torch.no_grad():
                    # # 太大了就用下面这个策略
                    # croped_imgs_lr = self.dst_eval.eval_crop(imgs_lr)
                    # croped_imgs_hr = self.dst_eval.eval_crop(imgs_hr)
                    # croped_imgs_dn = []
                    # for img_lr, img_hr in zip(croped_imgs_lr, croped_imgs_hr):
                    #     img_dn = self.net(img_lr)
                    #     croped_imgs_dn.append(img_dn)
                    # croped_imgs_dn = torch.cat(croped_imgs_dn)
                    # imgs_lr = self.dst_eval.eval_merge(croped_imgs_lr)
                    # imgs_dn = self.dst_eval.eval_merge(croped_imgs_dn)

                    # 扛得住就pad再crop
                    if imgs_lr.shape[-1] % 16 != 0:
                        p2d = (4,4,4,4)
                        imgs_lr = F.pad(imgs_lr, p2d, mode='reflect')
                        imgs_dn = self.net(imgs_lr)
                        imgs_lr = imgs_lr[..., 4:-4, 4:-4]
                        imgs_dn = imgs_dn[..., 4:-4, 4:-4]
                    else:
                        imgs_dn = self.net(imgs_lr)
                    
                    if 'vst' in self.dst['command']:
                        imgs_lr = inverse_VST_torch(imgs_lr, self.dst_eval.noiseparam, 
                            iso_list=data['ISO'], wp=self.dst['wp']-self.dst['bl'])
                        imgs_dn = inverse_VST_torch(imgs_dn, self.dst_eval.noiseparam, 
                            iso_list=data['ISO'], wp=self.dst['wp']-self.dst['bl'])
                    
                    # brighten
                    if self.dst['ori']:
                        imgs_lr = imgs_lr * ratio
                        imgs_dn = imgs_dn * ratio    
                    # if 'darkshading2' in self.dst['command']:
                    #     BLE = self.dst_eval.get_BLE_long(ISO, exp, self.dst_eval.naive) + self.dst_eval.blc_mean['kt_b_long'][ISO]
                    #     imgs_dn = imgs_dn + BLE / (16383 - 512)
                    # 只有在验证时才考虑这种骚操作
                    # self.args['brightness_correct'] = False
                    if self.args['brightness_correct'] and epoch < 0:
                        # ds_raw = self.dst_eval.get_darkshading(ISO)
                        # ds = raw2bayer(ds_raw, wp=16383-512, bl=0, norm=True, clip=False)
                        # imgs_ds = torch.from_numpy(ds[None,]).type(torch.FloatTensor).to(self.device)
                        # imgs_dn = imgs_dn + imgs_ds
                        imgs_dn = self.corrector(imgs_dn, imgs_hr)
                    imgs_lr = torch.clamp(imgs_lr, 0, 1)
                    imgs_dn = torch.clamp(imgs_dn, 0, 1)

                    # convert raw to rgb
                    if 'isp' in self.dst['command'].lower():
                        output = self.isp(imgs_dn)
                        output = output[0].clamp(0,1).detach().cpu().numpy().transpose(1,2,0)
                        output = np.uint8(output * 255)
                        # target = self.isp(imgs_hr)
                        # target = target[0].clamp(0,1).detach().cpu().numpy().transpose(1,2,0)
                        # target = np.uint8(target * 255)
                        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
                    else:
                        output = tensor2im(imgs_dn)
                        target = tensor2im(imgs_hr)
                    res = quality_assess(output, target, data_range=255)
                    raw_metrics = [res['PSNR'], res['SSIM']]
                    self.eval_psnr.update(res['PSNR'])
                    self.eval_ssim.update(res['SSIM'])
                    metrics[name] = raw_metrics
                    
                    if save_plot:
                        if self.infos is None:
                            inputs = tensor2im(imgs_lr)
                            res_in = quality_assess(inputs, target, data_range=255)
                            raw_metrics = [res_in['PSNR'], res_in['SSIM']] + raw_metrics
                        else:
                            raw_metrics = [self.infos[k]['PSNR_raw'], self.infos[k]['SSIM_raw']] + raw_metrics
                        if epoch > 0:
                            # self.multiprocess_plot(imgs_lr, imgs_dn, imgs_hr, 
                            #         wb, ccm, name, save_plot, epoch, raw_metrics, k)
                            pool.append(threading.Thread(target=self.multiprocess_plot, args=(imgs_lr, imgs_dn, imgs_hr, 
                                    wb, ccm, name, save_plot, epoch, raw_metrics, k)))
                            pool[k].start()
                        else:
                            infos = self.infos[k] if self.infos is not None else None
                            # 多进程
                            if infos is None:
                                inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
                                target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
                            else:
                                inputs = np.load(infos['path_npy_in'])
                                target = np.load(infos['path_npy_gt'])
                            if 'isp' not in self.dst['command'].lower():
                                output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
                            # raw_metrics = None # 用RGB metrics
                            task_list.append(
                                pool.submit(plot_sample, inputs, output, target, 
                                    filename=name, save_plot=save_plot, epoch=epoch,
                                    model_name=self.model_name, save_path=self.sample_dir,
                                    res=raw_metrics
                                    )
                                )

                    t.set_description(f'{name}')
                    t.set_postfix({'PSNR':f"{self.eval_psnr.avg:.2f}"})
                    t.update(1)

        if save_plot:
            if epoch > 0:
                for i in range(len(pool)):
                    pool[i].join()
            else:
                pool.shutdown(wait=True)
                for task in as_completed(task_list):
                    psnr, ssim, name = task.result()
                    metrics[name] = (psnr[1], ssim[1])
                    # if name[0] == '1' or self.dstname=='ELD':
                    self.eval_psnr_lr.update(psnr[0])
                    self.eval_psnr_dn.update(psnr[1])
                    self.eval_ssim_lr.update(ssim[0])
                    self.eval_ssim_dn.update(ssim[1])
        else:
            self.eval_psnr_dn = self.eval_psnr
            self.eval_ssim_dn = self.eval_ssim

        # 超过最好记录才保存
        if self.eval_psnr_dn.avg >= self.best_psnr and epoch > 0:
            self.best_psnr = self.eval_psnr_dn.avg
            log(f"Best PSNR is {self.best_psnr} now!!")
            model_dict = self.net.module.state_dict() if self.multi_gpu else self.net.state_dict()
            torch.save(model_dict, f'{self.fast_ckpt}/{self.model_name}_best_model.pth')

        log(f"Epoch {epoch}: PSNR={self.eval_psnr.avg:.2f}\n"
            +f"psnrs_lr={self.eval_psnr_lr.avg:.2f}, psnrs_dn={self.eval_psnr_dn.avg:.2f}"
            +f"\nssims_lr={self.eval_ssim_lr.avg:.4f}, ssims_dn={self.eval_ssim_dn.avg:.4f}",
            log=f'./logs/log_{self.model_name}.log')
        if epoch < 0:
            with open(metrics_path, 'wb') as f:
                pkl.dump(metrics, f)
        savefile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.jpg')
        logfile = os.path.join(self.sample_dir, f'{self.model_name}_eval_psnr.pkl')
        if epoch > 0:
            self.eval_psnr.plot_history(savefile=savefile, logfile=logfile)
        del pool
        plt.close('all')
        gc.collect()
        return metrics
    
    def multiprocess_plot(self, imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, raw_metrics, k):
        # if self.infos is None:
        inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
        # else:
        #     inputs = np.load(self.infos[k]['path_npy_in'])
        #     target = np.load(self.infos[k]['path_npy_gt'])
        output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
        
        psnr, ssim, _ = plot_sample(inputs, output, target, 
                        filename=name, 
                        save_plot=save_plot, epoch=epoch,
                        model_name=self.model_name,
                        save_path=self.sample_dir,
                        res=raw_metrics)
        self.eval_psnr_lr.update(psnr[0])
        self.eval_psnr_dn.update(psnr[1])
        self.eval_ssim_lr.update(ssim[0])
        self.eval_ssim_dn.update(ssim[1])

    def predict(self, raw, name='ds'):
        self.net.eval()
        img_lr = raw2bayer(raw+self.dst["bl"])[None, ...]
        img_lr = torch.from_numpy(img_lr)
        img_lr = img_lr.type(torch.FloatTensor).to(self.device)
        with torch.no_grad():
            croped_imgs_lr = self.dst_eval.eval_crop(img_lr)
            croped_imgs_dn = []
            for img_lr in tqdm(croped_imgs_lr):
                img_dn = self.net(img_lr)
                croped_imgs_dn.append(img_dn)
            croped_imgs_dn = torch.cat(croped_imgs_dn)
            img_dn = self.dst_eval.eval_merge(croped_imgs_dn)
            img_dn = img_dn
            img_dn = img_dn[0].detach().cpu().numpy()
        np.save(f'{name}.npy', img_dn)
    
    def preprocess(self, data, mode='train', preprocess=True):
        # 由于crops的存在，Dataloader会把数据变成6维，需要view回5维
        imgs_hr = tensor_dim5to4(data['hr']).type(torch.FloatTensor).to(self.device)
        imgs_lr = tensor_dim5to4(data['lr']).type(torch.FloatTensor).to(self.device)
        # self.use_gpu = True
        dst = self.dst_train if mode=='train' else self.dst_eval

        if self.use_gpu and mode=='train' and preprocess:
            b = imgs_lr.shape[0]
            if self.args['dst_train']['dataset'] == 'Proxy_Dataset':
                data['ratio'] = torch.ones(b, device=self.device)
                d = self.dst['patch_size']
                for i in range(b):
                    if 'params' not in self.dst:
                        noise_param = sample_params_proxy(camera_type=self.dst['camera_type'])
                    else:
                        noise_param = self.dst['params']
                    data['ratio'][i] = noise_param['ratio']
                    noise_param['shape'] = (4, 1, d, d)
                    proxy_net = self.PNNP_highISO if noise_param['ISO']>1600 else self.PNNP_lowISO
                    imgs_lr[i] = generate_noisy_proxy(proxy_net, y=imgs_lr[i], param=noise_param, 
                                    noise_code=self.dst['noise_code'], ori=self.dst['ori'])
            elif self.args['dst_train']['dataset'] == 'NF_Syn_Dataset':
                d = self.dst['patch_size']
                data['ratio'] = torch.ones(b, dtype=torch.float32, device=self.device)
                data['ISO'] = torch.ones(b, dtype=torch.float32, device=self.device)
                for i in range(b):
                    if 'params' not in self.dst:
                        noise_param = sample_params_proxy(camera_type=self.dst['camera_type'])
                    else:
                        noise_param = self.dst['params']
                    data['ratio'][i] = noise_param['ratio']
                    data['ISO'][i] = noise_param['ISO']
                    noise_param['shape'] = (1, 4, d, d)
                    proxy_net = self.proxy_net
                    imgs_lr[i] = generate_noisy_flow(proxy_net, y=imgs_lr[i], param=noise_param, 
                                    noise_code=self.dst['noise_code'], ori=self.dst['ori'])
            # BiSNA + PNNP
            elif self.args['dst_train']['dataset'] == 'PMNNP_Dataset':
                d = self.dst['patch_size']
                data['ratio'] = data['ratio'].view(-1).type(torch.FloatTensor).to(self.device)
                aug_r, aug_g, aug_b = get_aug_param_proxy(b=b, command=self.dst['command'])
                aug_wbs = torch.stack((aug_r, aug_g, aug_b, aug_g), dim=1)
                data['rgb_gain'] = torch.ones(b) * (aug_g + 1)
                data['wb'] = data['wb'][0].repeat(b, 1)
                for i in range(b):
                    aug_wb = aug_wbs[i].numpy()
                    if data['black_lr'][0]: aug_wb += 1
                    dgain = data['ratio'][i]
                    imgs_lr[i] = imgs_lr[i] if self.dst['ori'] else imgs_lr[i] * dgain
                    if np.abs(aug_wb).max() != 0:
                        data['wb'][i] *= (1+aug_wb[1]) / (1+aug_wb)
                        iso = data['ISO'][i//self.dst['crop_per_image']].item()
                        noise_param = sample_params_proxy_specificISO(camera_type=self.dst['camera_type'], iso=iso)
                        noise_param['exp'] = data['ExposureTime']
                        noise_param['ratio'] = dgain
                        noise_param['shape'] = (4, 1, d, d)
                        proxy_net = self.PNNP_highISO if noise_param['ISO']>1600 else self.PNNP_lowISO
                        noise_syn = generate_noisy_proxy(proxy_net, y=imgs_lr[i], param=noise_param, 
                                        noise_code=self.dst['noise_code'].replace('p',''), ori=self.dst['ori'])
                        imgs_lr[i], imgs_hr[i] = generate_PMNNP(imgs_lr[i], imgs_hr[i], noise_syn, aug_wb, ratio=dgain,
                                                                param=noise_param, ori=self.dst['ori'])
            else:
                log(f"No corresponding dataset (dst_train: {self.args['dst_train']['dataset']})?")
                raise NotImplementedError
        else: # mode == 'eval'
            pass
        
        ratio = data['ratio'].type(torch.FloatTensor).to(self.device)
        ratio = ratio.view(-1,1,1,1)
        if 'rgb_gain' in data:
            data['rgb_gain'] = data['rgb_gain'].type(torch.FloatTensor).to(self.device).view_as(ratio)
        
        if self.dst['clip']:
            lb = -np.inf if 'HB' in self.dst['command'] else 0
            imgs_lr = imgs_lr.clamp(lb, 1)
            imgs_hr = imgs_hr.clamp(0, 1)
        return imgs_lr, imgs_hr, ratio

    def load_proxynet(self):
        self.proxy = self.args['arch_proxy']
        self.proxy['noise_code'] = self.dst["noise_code"]
        self.proxy['ds_dir'] = self.dst["ds_dir"]
        # SonyA7S2 是双原生 ISO 传感器，这里按 lowISO/highISO 加载两段已训练 PNNP。
        # 公开版本只使用这些权重生成训练噪声，不包含 PNNP 本体训练过程。
        proxy_name = self.proxy['name']
        self.PNNP_lowISO = PNNP(self.proxy).requires_grad_(False)#globals()[self.arch_proxy['name']](self.arch_proxy)
        self.PNNP_highISO = PNNP(self.proxy).requires_grad_(False)
        # Loading Model
        proxy_model_lowISO = torch.load(f"{self.fast_ckpt}/{proxy_name}_lowISO.pth", map_location=self.device)
        proxy_model_highISO = torch.load(f"{self.fast_ckpt}/{proxy_name}_highISO.pth", map_location=self.device)
        self.PNNP_lowISO = load_weights(self.PNNP_lowISO, proxy_model_lowISO, by_name=True).to(self.device)
        self.PNNP_highISO = load_weights(self.PNNP_highISO, proxy_model_highISO, by_name=True).to(self.device)

        ds_k_lowISO = dataload(f'{self.proxy["ds_dir"]}/darkshading_lowISO_k.npy').astype(np.float32)
        ds_b_lowISO = dataload(f'{self.proxy["ds_dir"]}/darkshading_lowISO_b.npy').astype(np.float32)
        self.PNNP_lowISO.ds_k = torch.from_numpy(ds_k_lowISO).to(self.device)
        self.PNNP_lowISO.ds_b = torch.from_numpy(ds_b_lowISO).to(self.device)
        self.PNNP_lowISO.pack_ds()

        ds_k_highISO = dataload(f'{self.proxy["ds_dir"]}/darkshading_highISO_k.npy').astype(np.float32)
        ds_b_highISO = dataload(f'{self.proxy["ds_dir"]}/darkshading_highISO_b.npy').astype(np.float32)
        self.PNNP_highISO.ds_k = torch.from_numpy(ds_k_highISO).to(self.device)
        self.PNNP_highISO.ds_b = torch.from_numpy(ds_b_highISO).to(self.device)
        self.PNNP_highISO.pack_ds()
        # # AdaptiveDSC
        # self.proxy_dsc = AlphaDSC().to(self.device)
        # self.args['dst_train']['proxy_net'] = self.PNNP_highISO

def MultiProcessPlot(imgs_lr, imgs_dn, imgs_hr, wb, ccm, name, save_plot, epoch, 
                    raw_metrics, infos, model_name, sample_dir):
    if infos is None:
        inputs = raw2rgb_rawpy(imgs_lr, wb=wb, ccm=ccm)
        target = raw2rgb_rawpy(imgs_hr, wb=wb, ccm=ccm)
    else:
        inputs = np.load(infos['path_npy_in'])
        target = np.load(infos['path_npy_gt'])
    output = raw2rgb_rawpy(imgs_dn, wb=wb, ccm=ccm)
    
    psnr, ssim = plot_sample(inputs, output, target, 
                    filename=name, 
                    save_plot=save_plot, epoch=epoch,
                    model_name=model_name,
                    save_path=sample_dir,
                    res=raw_metrics)
    return psnr, ssim

class PNNP_SID_Parser():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def parse(self):
        self.parser.add_argument('--runfile', '-f', default="runfiles/SonyA7S2/PNNP_noiseflow.yml", type=Path, help="path to config")
        self.parser.add_argument('--mode', '-m', default='train', type=str, help="train or test")
        self.parser.add_argument('--debug', action='store_true', default=False, help="debug or not")
        self.parser.add_argument('--nofig', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--nohost', action='store_true', default=False, help="don't save_plot")
        self.parser.add_argument('--gpu', default="0", help="os.environ['CUDA_VISIBLE_DEVICES']")
        return self.parser.parse_args()

if __name__ == '__main__':
    trainer = PNNP_SID_Trainer()
    if trainer.mode == 'train':
        trainer.train()
        savefile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.jpg')
        logfile = os.path.join(trainer.sample_dir, f'{trainer.model_name}_train_psnr.pkl')
        trainer.train_psnr.plot_history(savefile=savefile, logfile=logfile)
        trainer.eval_psnr.plot_history(savefile=os.path.join(trainer.sample_dir, f'{trainer.model_name}_eval_psnr.jpg'))
        trainer.mode = 'evaltest'
    # best_model
    best_model_path = os.path.join(f'{trainer.fast_ckpt}/{trainer.model_name}_best_model.pth')
    if os.path.exists(best_model_path) is False: 
        best_model_path = os.path.join(f'{trainer.fast_ckpt}/{trainer.model_name}_last_model.pth')
    best_model = torch.load(best_model_path, map_location=trainer.device)
    trainer.net = load_weights(trainer.net, best_model)
    
    if 'dsc' in trainer.mode:
        dsc_model_path = os.path.join(f'{trainer.fast_ckpt}/{trainer.model_name}_dsc_model.pth')
        dsc_model = torch.load(dsc_model_path, map_location=trainer.device)
        trainer.proxy_dsc = load_weights(trainer.proxy_dsc, dsc_model, by_name=True)
        # SID
        # trainer.change_eval_dst('dsc')
        # metrics = trainer.train_dsc(-1)
        trainer.change_eval_dst('test')
        SID_ratio_list = [100,250,300]
        for dgain in SID_ratio_list:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'SID Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            metrics = trainer.train_dsc(-1)

    if 'eval' in trainer.mode:
        # ELD
        trainer.change_eval_dst('eval')
        for dgain in trainer.args['dst_eval']['ratio_list']:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'ELD Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.ratio_list=[dgain]
            trainer.dst_eval.recheck_length()
            metrics = trainer.eval(-1)

    if 'test' in trainer.mode:
        # SID
        trainer.change_eval_dst('test')
        SID_ratio_list = [100, 250, 300]
        for dgain in SID_ratio_list:
            info_path = os.path.join(trainer.cache_dir, f'{trainer.dstname}_{dgain}.pkl')
            if os.path.exists(info_path):
                with open(info_path,'rb') as f:
                    trainer.infos = pkl.load(f)
            log(f'SID Datasets: Dgain={dgain}',log=f'./logs/log_{trainer.model_name}.log')
            trainer.dst_eval.change_eval_ratio(ratio=dgain)
            metrics = trainer.eval(-1)
    log(f'Metrics have been saved in ./metrics/{trainer.model_name}_metrics.pkl')
