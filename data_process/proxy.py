from utils import *

"""PNNP 开源训练用噪声合成工具。

这里不训练 PNNP 本体，只负责：
1. 根据相机类型采样 ISO、曝光时间、系统增益和读噪声参数；
2. 调用已发布的 PNNP 权重生成信号无关读噪声；
3. 叠加 shot noise、行列噪声、量化噪声等物理分量，形成下游去噪训练样本。
"""

def get_camera_proxy_params(camera_type=None):
    cam_noisy_params = {}
    cam_noisy_params['IMX686'] = { # ISO-6400
        'K_k':0.001366021, 'K_b':0, 'lam':0.102, 'q':1/(2**10), 'wp':1023, 'bl':64,
        'ISOmin':6400, 'ISOmax':6400, 'tmin':1, 'tmax':40, # exposure time(ms)
        # 'sigTLk':0.85187, 'sigTLb':0.07991,   'sigTLsig':0.02921, # 有bug, 待重测
        'sigRk':0.87611,  'sigRb':-2.11455,   'sigRsig':0.03274,
        'sigGsk':0.85187, 'sigGsb':0.67991,   'sigGssig':0.02921,
    }
    cam_noisy_params['SonyA7S2_lowISO'] = {
        'K_k':0.0009546, 'K_b':-0.00193, 'lam':-0.026, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'ISOmin':50, 'ISOmax':1600, 'tmin':10, 'tmax':100, # exposure time(ms)
        'sigRk':0.78782,  'sigRb':-0.34227,  'sigRsig':0.02832,
        'sigTLk':0.74043, 'sigTLb':0.86182, 'sigTLsig':0.00712,
        'sigGsk':0.82966, 'sigGsb':1.49343, 'sigGssig':0.00359,
        'sigReadk':0.82879, 'sigReadb':1.50601, 'sigReadsig':0.00362,
        'uReadk':0.01472, 'uReadb':0.01129, 'uReadsig':0.00034,
    }
    cam_noisy_params['SonyA7S2_highISO'] = { # ISO-2000 ~ ISO-3200
        'K_k':0.0009546, 'K_b':-0.00193, 'lam':-0.025, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'ISOmin':2000, 'ISOmax':25600, 'tmin':10, 'tmax':100, # exposure time(ms)
        'sigRk':0.62945,  'sigRb':-1.51040,  'sigRsig':0.02609,
        'sigTLk':0.74901, 'sigTLb':-0.12348, 'sigTLsig':0.00638,
        'sigGsk':0.82878, 'sigGsb':0.44162, 'sigGssig':0.00153,
        'sigReadk':0.82645, 'sigReadb':0.45061, 'sigReadsig':0.00156,
        'uReadk':0.00385, 'uReadb':0.00674, 'uReadsig':0.00039,
    }
    cam_noisy_params['CRVD'] = {
        'K_k':9.7681e-4, 'K_b':2.1558621, 'lam':0.015, 'q':1/(2**12), 'wp':4095, 'bl':240,
        'ISOmin':1600, 'ISOmax':25600, 'tmin':30, 'tmax':30, # exposure time(ms)
        'sigRk':0.93368,  'sigRb':-2.19692,  'sigRsig':0.02473,
        'sigGsk':0.95387, 'sigGsb':0.01552, 'sigGssig':0.00855,
        'sigTLk':0.95495, 'sigTLb':0.01618, 'sigTLsig':0.00790,
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use IMX686's parameters to test.''')
        return cam_noisy_params['IMX686']

# 噪声参数采样
def sample_params_proxy(camera_type='SonyA7S2', ln_ratio=False):
    choice = 1
    Dual_ISO_Cameras = ['SonyA7S2']
    if camera_type in Dual_ISO_Cameras:
        choice = np.random.randint(2)
        camera_type += '_lowISO' if choice<1 else '_highISO'

    # 获取已经测算好的相机噪声参数（范围）
    p = get_camera_proxy_params(camera_type=camera_type)
    wp = p['wp']
    bl = p['bl']
    q = p['q']
    lam = p['lam']
    
    # 采样相机噪声参数(ISO, 曝光时间t_exp)
    log_ISO = np.random.uniform(low=np.log(p['ISOmin']), high=np.log(p['ISOmax']))
    log_t = np.random.uniform(low=np.log(p['tmin']), high=np.log(p['tmax']))
    iso = np.exp(log_ISO)
    t = np.exp(log_t)
    K = p['K_k'] * iso - p['K_b'] # SonyA7S2
    log_K = np.log(K)
    mu_TL = p['sigTLk']*log_K + p['sigTLb'] if 'sigTLk' in p else q
    mu_R = p['sigRk']*log_K + p['sigRb'] if 'sigRk' in p else q
    mu_Gs = p['sigGsk']*log_K + p['sigGsb'] if 'sigGsk' in p else q
    log_sigTL = np.random.normal(loc=mu_TL, scale=p['sigTLsig']) if 'sigTLk' in p else 0
    log_sigR = np.random.normal(loc=mu_R, scale=p['sigRsig']) if 'sigRk' in p else 0
    log_sigGs = np.random.normal(loc=mu_Gs, scale=p['sigGssig']) if 'sigGsk' in p else q
    # 去掉log
    sigTL = np.exp(log_sigTL)
    sigR = np.exp(log_sigR)
    sigGs = np.exp(log_sigGs)
    bias = p['bias'] if 'bias' in p else 0

    # 模拟曝光衰减的系数, ln_ratio模式会照顾弱噪声场景, 更有通用性
    if ln_ratio:
        high = 1 if 'CRVD' in camera_type else 5
        log_ratio = np.random.uniform(low=-0.01, high=high)
        ratio = np.exp(log_ratio) # np.random.uniform(low=1, high=200) if choice else np.exp(log_ratio)
    else:
        ratio = np.random.uniform(low=1, high=16) if camera_type=='IMX686' else np.random.uniform(low=100, high=300)
        
    return {'K':K, 'ISO':iso, 'exp':t, 'sigTL':sigTL, 'sigR':sigR, 'sigGs':sigGs,
            'lam':lam,'q':q, 'ratio':ratio, 'wp':wp, 'bl':bl, 'bias':bias}

# 噪声参数采样
def sample_params_proxy_specificISO(camera_type='SonyA7S2', iso=1600):
    Dual_ISO_Cameras = ['SonyA7S2']
    if camera_type in Dual_ISO_Cameras:
        choice = 0 if iso<=1600 else 1
        camera_type += '_lowISO' if choice<1 else '_highISO'

    # 获取已经测算好的相机噪声参数（范围）
    p = get_camera_proxy_params(camera_type=camera_type)
    wp = p['wp']
    bl = p['bl']
    q = p['q']
    lam = p['lam']
    
    # 采样相机噪声参数(ISO, 曝光时间t_exp)
    t = 100
    K = p['K_k'] * iso - p['K_b'] # SonyA7S2
    log_K = np.log(K)
    mu_TL = p['sigTLk']*log_K + p['sigTLb'] if 'sigTLk' in p else q
    mu_R = p['sigRk']*log_K + p['sigRb'] if 'sigRk' in p else q
    mu_Gs = p['sigGsk']*log_K + p['sigGsb'] if 'sigGsk' in p else q
    log_sigTL = mu_TL
    log_sigR = mu_R
    log_sigGs = mu_Gs
    # 去掉log
    sigTL = np.exp(log_sigTL)
    sigR = np.exp(log_sigR)
    sigGs = np.exp(log_sigGs)
    bias = p['bias'] if 'bias' in p else 0
    ratio = 100
        
    return {'K':K, 'ISO':iso, 'exp':t, 'sigTL':sigTL, 'sigR':sigR, 'sigGs':sigGs,
            'lam':lam,'q':q, 'ratio':ratio, 'wp':wp, 'bl':bl, 'bias':bias}

def generate_noisy_proxy(proxy_net, y=None, camera_type=None,  noise_code='p', param=None, ori=False, clip=False):
    """使用 PNNP 权重为干净 raw patch 合成低光噪声。

    Args:
        proxy_net: 已加载权重的 PNNP，Sony 为 low/high ISO，IMX686 为 cold/hot 分段。
        y: 四通道 Bayer 干净图，范围通常为 [0, 1]。
        noise_code: 控制叠加哪些物理分量；常用 `pr` 表示 shot noise + 行列读噪声。
        ori: False 时会把短曝光图线性提亮到长曝光亮度，匹配去噪训练输入。
    """
    p = param if param is not None else get_camera_proxy_params(camera_type)
    p['shape'] = (4,1,y.shape[-2],y.shape[-1])
    p['device'] = y.device
    # mode detect
    noise_code = noise_code.lower()
    use_R = True if 'r' in noise_code else False
    use_Q = True if 'q' in noise_code else False
    use_TL = True if 'tl' in noise_code else False
    use_G = True if 'g' in noise_code else False
    use_P = True if 'p' in noise_code else False
    use_D = True if 'd' in noise_code else False
    use_Joint = True #if 'j' in noise_code else False
    ## 以下内容均为物理噪声模型, 体现“双路可选”的思想
    if y is not None:
        if len(y.shape) == 3: y = y.unsqueeze(0)
        y = y * (p['wp'] - p['bl'])
        y = y / p['ratio']
        if use_P:   # 使用泊松噪声作为shot noise
            K = p['K']
            N_p = tdist.Poisson(y/K).sample() * K
            p['clean'] = None
        else:   # 不考虑shot noise
            N_p = 0
            p['clean'] = y
    if use_TL:   # 使用TL噪声作为read noisy
        N_read = torch.from_numpy(stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'], size=y.shape).astype(np.float32))
    elif use_G:   # 使用高斯噪声作为read noisy
        N_read = tdist.Normal(loc=torch.zeros_like(y), scale=p['sigGs']).sample()
    else:
        N_read = 0
    # 行噪声
    if use_R:
        # 因为此处是rgbg, 每两个是一行
        N_fpnrow, N_fpncol = 0.25, 0.15
        if use_Joint and torch.rand(1) < 0.1:
            N_fpnrow = N_fpnrow * 10
            N_fpncol = N_fpncol * 10
        N_r1 = torch.randn((y.shape[0], 1, y.shape[2], 1), device=p['device']) * p['sigR']
        N_r2 = torch.randn((y.shape[0], 1, y.shape[2], 1), device=p['device']) * p['sigR']
        N_r = torch.cat((N_r1, N_r1, N_r2, N_r2), dim=1) * (1 + N_fpnrow)
        N_c1 = torch.randn((y.shape[0], 1, 1, y.shape[3]), device=p['device']) * p['sigR']
        N_c2 = torch.randn((y.shape[0], 1, 1, y.shape[3]), device=p['device']) * p['sigR']
        N_c = torch.cat((N_c1, N_c2, N_c2, N_c1), dim=1) * N_fpncol
    else:
        N_r = N_c = 0
    # 量化噪声
    N_q = (torch.rand(y.shape, device=p['device']) - 0.5) * p['q'] * (p['wp'] - p['bl']) if use_Q else 0
    # 偏置项
    # N_bias = torch.from_numpy(p['bias'].reshape(1,-1,1,1)) if use_D else 0
    N_bias = torch.randn((1,1,1,1), dtype=torch.float32, device=p['device']) * 0.002 * p['ISO']**0.5 # 标定经验统计规律
    if 'SonyA7S2' in camera_type:
        if p['ISO'] > 1600: N_bias *= 2
        if use_Joint and torch.rand(1) < 0.1:
            N_bias = N_bias * 10
    # PNNP 负责生成最难用解析模型描述的像素级读噪声。
    with torch.no_grad():
        N_proxy = proxy_net.sample(p)
        N_proxy = N_proxy.permute(1,0,2,3) #reshape(*y.shape)

    # 归一化回[0, 1]
    x = (N_proxy + N_p + N_read + N_r + N_c + N_q + N_bias) / (p['wp'] - p['bl'])
    # 模拟实际raw的clip情况
    x = torch.clamp(x, 0, 1) if clip is True else torch.clamp(x, -p['bl']/p['wp'], 1)
    # ori_brightness
    if ori is False:
        x = x * p['ratio']

    return x.squeeze(0)

def generate_noisy_flow(proxy_net, y=None, camera_type=None,  noise_code='p', param=None, ori=False, clip=False):
    p = param if param is not None else get_camera_proxy_params(camera_type)
    p['device'] = y.device
    # mode detect
    noise_code = noise_code.lower()
    use_R = True if 'r' in noise_code else False
    use_Q = True if 'q' in noise_code else False
    use_TL = True if 'tl' in noise_code else False
    use_G = True if 'g' in noise_code else False
    use_P = True if 'p' in noise_code else False
    use_D = True if 'd' in noise_code else False
    use_Joint = True #if 'j' in noise_code else False
    ## 以下内容均为物理噪声模型, 体现“双路可选”的思想
    if y is not None:
        if len(y.shape) == 3: y = y.unsqueeze(0)
        y = y * (p['wp'] - p['bl'])
        y = y / p['ratio']
        if use_P:   # 使用泊松噪声作为shot noise
            K = p['K']
            N_p = tdist.Poisson(y/K).sample() * K
            p['clean'] = None
        else:   # 不考虑shot noise
            N_p = 0
            p['clean'] = y
    else:
        raise NotImplementedError
    if use_TL:   # 使用TL噪声作为read noisy
        N_read = torch.from_numpy(stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'], size=y.shape).astype(np.float32))
    elif use_G:   # 使用高斯噪声作为read noisy
        N_read = tdist.Normal(loc=torch.zeros_like(y), scale=p['sigGs']).sample()
    else:
        N_read = 0
    # 行噪声
    if use_R:
        # 因为此处是rgbg, 每两个是一行
        N_fpnrow, N_fpncol = 0,0#0.25, 0.15
        if use_Joint and torch.rand(1) < 0.1:
            N_fpnrow = N_fpnrow * 10
            N_fpncol = N_fpncol * 10
        N_r1 = torch.randn((y.shape[0], 1, y.shape[2], 1), device=p['device']) * p['sigR']
        N_r2 = torch.randn((y.shape[0], 1, y.shape[2], 1), device=p['device']) * p['sigR']
        N_r = torch.cat((N_r1, N_r1, N_r2, N_r2), dim=1) * (1 + N_fpnrow)
        N_c1 = torch.randn((y.shape[0], 1, 1, y.shape[3]), device=p['device']) * p['sigR']
        N_c2 = torch.randn((y.shape[0], 1, 1, y.shape[3]), device=p['device']) * p['sigR']
        N_c = torch.cat((N_c1, N_c2, N_c2, N_c1), dim=1) * N_fpncol
    else:
        N_r = N_c = 0
    # 量化噪声
    N_q = (torch.rand(y.shape, device=p['device']) - 0.5) * p['q'] * (p['wp'] - p['bl']) if use_Q else 0
    # 偏置项
    # N_bias = torch.from_numpy(p['bias'].reshape(1,-1,1,1)) if use_D else 0
    N_bias = torch.randn((1,1,1,1), dtype=torch.float32, device=p['device']) * 0.001 * p['ISO']**0.5 # 标定统计规律
    if p['ISO'] > 1600: N_bias *= 2
    # if use_Joint and torch.rand(1) < 0.1:
    #     N_bias = N_bias * 10
    # Proxy Net！！
    with torch.no_grad():
        kwargs = {
            'noise': torch.randn_like(y),
            'iso': torch.tensor(p['ISO'], dtype=torch.float32, device=p['device']),
        }
        N_proxy = proxy_net.sample(**kwargs)

    # 归一化回[0, 1]
    x = (N_proxy + N_p + N_read + N_r + N_c + N_q + N_bias) / (p['wp'] - p['bl'])
    # 模拟实际raw的clip情况
    x = torch.clamp(x, 0, 1) if clip is True else torch.clamp(x, -p['bl']/p['wp'], 1)
    # ori_brightness
    if ori is False:
        x = x * p['ratio']

    return x.squeeze(0)

def get_aug_param_proxy(b=8, command='augv2', numpy=False, camera_type='SonyA7S2'):
    aug_r, aug_g, aug_b = torch.zeros(b), torch.zeros(b), torch.zeros(b)
    r = np.random.randint(2) * 0.25 + 0.25
    u = r
    if np.random.randint(4):
        if 'augv2' in command:
            aug_g = torch.clamp(torch.randn(b) * r, -0.5, 4*u)
            aug_r = torch.clamp((1+torch.randn(b)*r) * (1+aug_g) - 1, 1/(1+4*u) - 1, 4*u)
            aug_b = torch.clamp((1+torch.randn(b)*r) * (1+aug_g) - 1, 1/(1+4*u) - 1, 4*u)
        else:
            raise NotImplementedError
    if numpy:
        aug_r = np.squeeze(aug_r.numpy())
        aug_g = np.squeeze(aug_g.numpy())
        aug_b = np.squeeze(aug_b.numpy())
    return aug_r, aug_g, aug_b

def generate_PMNNP(img_lr, img_hr, noise_syn, aug_wb, param=None, ratio=100, ori=False):
    '''原理
    1. read noise和ISO绑定, 因此在【增广】中, 理论上不会发生变化。
    2. shot noise的变化和aug_wb系数绑定:
        a) aug_wb均为正数时, 等效于SNA
        b) aug_wb存在负数时, 触发BiSNA+PNNP
    3. BiSNA+PNNP仅对齐噪声的统计量, 即标准差。设scale系数为a
        a) shot noise需要用泊松分布补齐方差, 即方差为a-a**2的纯噪声
        b) read noise需要用PNNP补齐完整的读噪声, 即方差为1-a**2的纯噪声
        c) shot noise按照SNA的逻辑继续正向增广
    '''
    p = param
    p['shape'] = (4,1,img_hr.shape[-2],img_hr.shape[-1])
    p['device'] = img_hr.device

    if aug_wb is not None:
        # 默认pattern为RGGB！(通道rgbg排列)
        gt = img_hr * (p['wp'] - p['bl']) / ratio
        noisy = img_lr * (p['wp'] - p['bl']) / ratio
        # 补噪声
        daug = -np.minimum(np.min(aug_wb), 0)
        daug = torch.from_numpy(np.array(daug)).to(gt.device)
        aug_wb = torch.from_numpy(aug_wb).to(gt.device)
        dy = gt * aug_wb.reshape(-1,1,1)    # 不量化对多样性更友好
        if daug == 0:
            # 只有增益的话很好处理，叠加泊松分布就行
            dn = tdist.Poisson(dy/p['K']).sample() * p['K']
        else:
            # warnings.warn('You are using BiSNA!!!')
            # 存在减益的话就很麻烦，需要考虑read noise并且补齐分布
            scale = 1 - daug
            # 要保证dyn是非负的
            aug_wb_new = aug_wb + daug
            dyn = gt * aug_wb_new.reshape(-1,1,1)
            # 先通过缩放减小噪声图
            noisy *= scale
            # 补齐单个照片的读噪声
            dn_read = noise_syn * torch.sqrt(1-scale**2)
            # 补齐由于除法导致的分布变化，恢复shot noise应有的分布
            ds = scale - scale**2
            # dn_shot = tdist.Poisson(ds * gt/p['K']).sample() * p['K'] - gt * ds
            dn_shot = 0 # 加速，和下面的Poisson合并
            # 叠加泊松分布
            dn_aug = tdist.Poisson((ds*gt + dyn)/p['K']).sample() * p['K'] - ds*gt
            dn = dn_read + dn_shot + dn_aug
        # 归一化回[0, 1]
        gt = torch.clamp((gt + dy)*ratio, 0, (p['wp'] - p['bl']))
        noisy = torch.clamp(noisy + dn, -p['bl'], (p['wp'] - p['bl']))
        gt /= (p['wp'] - p['bl'])
        noisy /= (p['wp'] - p['bl'])

    if ori is False:
        noisy *= ratio

    return noisy.squeeze(0), gt.squeeze(0)