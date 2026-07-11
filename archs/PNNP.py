# utf-8
from .modules import *

def hook_plot(dists, params={}):
    from matplotlib import pyplot as plt
    import scipy.stats
    n = len(dists)
    print(params)
    fig = plt.figure(figsize=(n*5,4))
    axs = [None] * n
    for i in range(n):
        axs[i] = plt.subplot(1,n,i+1)
        data = dists[i].view(-1).cpu().detach().numpy()
        print(data.std())
        scipy.stats.probplot(data, plot=axs[i], dist='norm', rvalue=True)

def norm(data, bl=None, wp=None, clip=False):
    data = data.astype(np.float32)
    if clip and wp is not None:
        data = data.clip(-bl, wp)
    bl = data.min() if bl is None else bl
    wp = data.max() if wp is None else wp
    return (data - bl) / (wp - bl)

def normalize(data):
    mu, sig = data.mean(dim=-1), data.std(dim=-1)
    data = (data-mu) / sig
    return data, mu, sig

def inv_normalize(data, mu, sig):
    return data * sig + mu

class CDFPPF(torch.nn.Module):
    def __init__(self, data, inf=None):
        super().__init__()
        self.sorted_data, _ = torch.sort(data)
        inf = torch.tensor([torch.inf,]) if inf is None else torch.tensor([inf,])
        inf = inf.to(data.device)
        self.sorted_data_pad = torch.cat((-inf, self.sorted_data))
        # self.cdf = torch.linspace(0., 1., len(data))

    def cdf_interp(self, x):
        idx = torch.searchsorted(self.sorted_data_pad, x)
        w = self.sorted_data_pad[idx] - x
        diff = self.sorted_data_pad[idx] - self.sorted_data_pad[idx-1]
        delta = w / diff
        idx_interp = idx - delta
        cdf_interp = (idx_interp-1) / (len(self.sorted_data_pad) - 2)
        return cdf_interp

    def get_cdf(self, x):
        x = torch.clamp(x, self.sorted_data[0], self.sorted_data[-1])
        # CDF计算
        # idx = torch.searchsorted(self.sorted_data, x)
        # cdf = idx.float() / (len(self.sorted_data) - 1)
        cdf = self.cdf_interp(x)
        return cdf

# Quantile Loss
def QuantileLoss(output, gt, x_quant):
    qout = torch.quantile(output, x_quant, dim=-1, keepdim=False, interpolation='linear').squeeze()
    qgt = torch.quantile(gt, x_quant, dim=-1, keepdim=False, interpolation='linear').squeeze()
    qt_loss = F.l1_loss(qout, qgt, reduction='mean')
    return qt_loss

# CDF Loss
def CDFLoss(output, gt, x_cdf):
    cdfout = CDFPPF(output.view(-1)).get_cdf(x_cdf)
    cdfgt= CDFPPF(gt.view(-1)).get_cdf(x_cdf)
    cdf_loss = F.l1_loss(cdfout, cdfgt, reduction='mean')
    return cdf_loss

def KLD(output, gt, x_pdf):
    q = cdf2pdf(CDFPPF(output.view(-1)).get_cdf(x_pdf)).clamp_min_(1e-9)
    p = cdf2pdf(CDFPPF(gt.view(-1)).get_cdf(x_pdf)).clamp_min_(1e-9)
    factor = torch.max(q.sum(), p.sum()).detach()
    q = q / factor
    p = p / factor
    # idx = (q > 0) & (p > 0)
    # p = p[idx]
    # q = q[idx]
    logp = torch.log(p)
    logq = torch.log(q)
    kl_loss = torch.sum(p * (logp - logq))
    return kl_loss

def cdf2pdf(data):
    diff_kernel = torch.tensor([1,-1], dtype=torch.float32).to(data.device).view(1,1,-1)
    return torch.abs(F.conv1d(data.view(1,1,-1), diff_kernel)).view(-1)

def get_x(sigma=4, size=1000, mode='uniform', random=True):
    x = torch.linspace(10**(-sigma), 1-10**(-sigma), size)
    if mode == 'uniform':
        return x
    elif mode == 'cdf':
        norm = torch.distributions.Normal(loc=0,scale=1)
        x = norm.cdf(x*sigma*2-sigma)
        if random:
            eps = (torch.randn(size)) / 10 ** sigma
            x = torch.clamp(x + eps, 0, 1)
    elif mode == 'icdf':
        norm = torch.distributions.Normal(loc=0,scale=1)
        # x = norm.icdf(x)
        if random:
            # eps = (torch.rand(size) - 0.5) / 10 ** sigma
            eps = (torch.randn(size)/2) / 10 ** sigma
            x = x + eps
        x = norm.icdf(x.clamp(10**(-sigma), 1-10**(-sigma)))
    return torch.sort(x)[0]

class ProxyNet_Base(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nf=16, nb=2, mode='2stage', act='swish'):
        super().__init__()
        self.mode = mode
        self.act = nn.SiLU() if act=='swish' else nn.Identity()
        k = 1 if 'conv3x3' not in self.mode else 3
        # ISO 相关分支：输出会乘以线性/分段线性增益 K(ISO)。
        self.nn_pre = nn.Sequential(
            nn.Conv2d(in_nc, nf, k, 1, padding='same'),
            self.act,
            *[ResBlock_Dist(nf, k, act) for i in range(nb)],
            nn.Conv2d(nf, out_nc, k, 1, padding='same')
        )
        # ISO 无关分支：用于表达不随模拟增益线性变化的残余读噪声。
        self.nn_fol = nn.Sequential(
            nn.Conv2d(in_nc, nf, k, 1, padding='same'),
            self.act,
            *[ResBlock_Dist(nf, k, act) for i in range(nb)],
            nn.Conv2d(nf, out_nc, k, 1, padding='same')
        )
        if '1stage' in self.mode:
            self.nn_trans = nn.Sequential(
                nn.Conv2d(in_nc+1, nf, k, 1, padding='same'),
                self.act,
                *[ResBlock_Dist(nf, 1, act) for i in range(nb+1)],
                nn.Conv2d(nf, out_nc, k, 1, padding='same')
            )
            self.out = nn.Conv2d(nf, out_nc, k, 1, padding='same')
        
        # scaler 是原训练中学习到的 ISO 分段校正项；开源训练只加载已发布权重并推理。
        self.legal_iso = torch.tensor([50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600] +\
            [2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000, 51200], dtype=torch.float32)
        self.ISO2K = nn.Parameter(torch.tensor((0.0009546, -0.00193), dtype=torch.float32), requires_grad=False)
        self.scaler = nn.Parameter(torch.tensor(np.zeros_like(self.legal_iso), dtype=torch.float32), requires_grad=True)
        self.trigger = False
        self.cache = {}
    
    def _scale(self, iso):
        l = torch.searchsorted(self.legal_iso, iso, right=False)
        r = l + 1
        if r < self.legal_iso.shape[0]:
            iso_l, iso_r = self.legal_iso[l], self.legal_iso[r]
            scale_l, scale_r = torch.exp(self.scaler[l]), torch.exp(self.scaler[r])
            scale = ((iso - iso_l) * scale_r + (iso_r - iso) * scale_l) / (iso_r - iso_l)
        else:
            scale = torch.exp(self.scaler[l])
        return scale

    def get_Nproxy(self, x_pre, x_fol, iso):
        trans_pre = self.nn_pre(x_pre)
        trans_fol = self.nn_fol(x_fol)
        K = self.ISO2K[0] * iso + self.ISO2K[1]
        K = K * self._scale(iso)
        out = trans_pre * K + trans_fol
        if self.trigger:
            self.trigger = False
            hook_plot(dists=(trans_pre, trans_fol), params={'K',K})
        return out
    
    def get_Nproxy_1class(self, x, iso):
        K = self.ISO2K[0] * iso + self.ISO2K[1]
        K = K * self._scale(iso) + (x * 0)
        inp = torch.cat((x, K), dim=1)
        trans_x = self.nn_trans(inp)
        out = trans_x
        return out
    
    def get_Nphysics(self, p):
        if 'g' in p['noise_code']:
            out = torch.distributions.Normal(loc=torch.zeros(p['shape']), scale=p['sigma']).sample()
        else:
            raise NotImplementedError
        return out

    def forward(self, p, x_pre=None, x_fol=None):
        if self.mode == '1stage':
            x = x_pre if x_pre is not None else torch.randn(p['shape'], dtype=torch.float32, device=p['device'])
            out = self.get_Nproxy_1class(x, p['ISO'])
        elif self.mode== 'physics':
            out = self.get_Nphysics(p)
        else:
            x_pre = x_pre if x_pre is not None else torch.randn(p['shape'], dtype=torch.float32, device=p['device'])
            x_fol = x_fol if x_fol is not None else torch.randn(p['shape'], dtype=torch.float32, device=p['device'])
            out = self.get_Nproxy(x_pre, x_fol, p['ISO'])
        return out

class ProxyNet_Pixel(ProxyNet_Base):
    def __init__(self, in_nc=1, out_nc=1, nf=16, nb=2, mode='2stage', act='swish'):
        super().__init__(in_nc, out_nc, nf, nb, mode, act)
    
    def get_Nphysics(self, p):
        if 'g' in p['noise_code']:
            out = torch.distributions.Normal(loc=torch.zeros(p['shape']), scale=p['sigGs']).sample()
        else:
            raise NotImplementedError
        return out
    
    def forward(self, p, x_pre=None, x_fol=None):
        return super().forward(p, x_pre, x_fol)

class ProxyNet_Band(ProxyNet_Base):
    def __init__(self, in_nc=1, out_nc=1, nf=16, nb=1, mode='2stage', act='swish'):
        super().__init__(in_nc, out_nc, nf, nb, mode, act)
    
    def get_Nphysics(self, p):
        if 'r' in p['noise_code']:
            out = torch.distributions.Normal(loc=torch.zeros((p['B'],p['C'],p['H'],1)), scale=p['sigR']).sample()
        else:
            raise NotImplementedError
        return out
    
    def forward(self, p, x_pre=None, x_fol=None):
        return super().forward(p, x_pre, x_fol)

# Physics-guided Noise Neural Proxy
class PNNP(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.default_args(args)
        in_nc, out_nc, nf, nb = self.args['in_nc'], self.args['out_nc'], self.args['nf'], self.args['nb']
        self.mode = self.args['mode']
        self.ds_k = None
        self.ds_b = None
        self.H, self.W = self.args['H'] // 2, self.args['W'] // 2
        self.shape = (4, 1, self.H, self.W)
        print('Mode:', self.mode)
        # 公开版本只用于加载已训练 PNNP 权重并采样噪声；PNNP 本体训练损失未开放。
        self.noise_code = self.args['noise_code'].lower()
        self.use_Frame = True if 'f' not in self.noise_code else False
        self.use_Band = True if 'r' not in self.noise_code else False
        self.use_Pixel = True if 'g' not in self.noise_code else False
        if 'wopnd' in self.mode:
            self.use_Frame = False
            self.use_Band = False
        # band-wise noise proxy
        print(f'Frame-wise noise proxy: {self.use_Frame}')
        print(f'Band-wise noise proxy: {self.use_Band}')
        print(f'Pixel-wise noise proxy: {self.use_Pixel}')
        if self.use_Band:
            self.proxy_row = ProxyNet_Band(in_nc, out_nc, nf, nb//2, self.mode, act='swish')
            self.proxy_col = ProxyNet_Band(in_nc, out_nc, nf, nb//2, self.mode, act='swish')
        else:
            self.proxy_row = self.proxy_col = None
        # pixel-wise noise proxy
        self.proxy_pixel = ProxyNet_Pixel(in_nc, out_nc, nf, nb, self.mode, act=self.args['act']) if self.use_Pixel else None
        self.ISO2K_require_grad(False)
    
    def default_args(self, args=None):
        self.args = {}
        self.args['in_nc'] = 1
        self.args['out_nc'] = 1
        self.args['H'] = 2848
        self.args['W'] = 4256
        self.args['nf'] = 16
        self.args['nb'] = 2
        self.args['nframes'] = 1
        self.args['d'] = 1024
        self.args['mode'] = '2stage+iso'
        self.args['noise_code'] = 'r'
        self.args['act'] = 'swish'
        if args is not None:
            for key in args:
                self.args[key] = args[key]

    def pack_ds(self):
        """将 Bayer 排列的 dark-shading 标定图打包成 RGBG 四通道，供 patch 采样使用。"""
        ds_k = torch.empty(self.shape, dtype=torch.float32, device=self.ds_k.device)
        ds_b = torch.empty(self.shape, dtype=torch.float32, device=self.ds_k.device)
        ds_k[0,0], ds_k[1,0], ds_k[2,0], ds_k[3,0] = self.ds_k[0::2,0::2], self.ds_k[0::2,1::2], self.ds_k[1::2,1::2], self.ds_k[1::2,0::2]
        ds_b[0,0], ds_b[1,0], ds_b[2,0], ds_b[3,0] = self.ds_b[0::2,0::2], self.ds_b[0::2,1::2], self.ds_b[1::2,1::2], self.ds_b[1::2,0::2]
        self.ds_k = ds_k
        self.ds_b = ds_b
    
    def ISO2K_require_grad(self, mode=True, ISO2K=None):
        """设置 ISO->系统增益线性层是否可学习；公开训练中固定并加载 checkpoint。"""
        print(f'Set ISO2K learnable: {mode}')
        if self.proxy_pixel is None:
            return
        self.proxy_pixel.ISO2K.requires_grad_(mode)
        if ISO2K is not None:
            self.proxy_pixel.ISO2K[0].fill_(ISO2K[0])
            self.proxy_pixel.ISO2K[1].fill_(ISO2K[1])

    def loss(self, p=None):
        """PNNP 本体训练损失未开放，公开版本仅用于加载已训练 PNNP 权重并采样噪声。"""
        raise NotImplementedError
    def forward(self, p=None, inputs={}):
        x_pre = inputs['x_pre'] if 'x_pre' in inputs else None
        x_fol = inputs['x_fol'] if 'x_fol' in inputs else None
        # Pixel-wise Noise
        noise = self.proxy_pixel(p, x_pre, x_fol)
        return noise

    @torch.no_grad()
    def sample(self, p):
        """采样 PNNP 读噪声。

        输入 p 由 data_process/proxy.py 组装，包含 ISO、曝光时间、输出 shape 和 device。
        下游训练会再叠加 shot noise、行列噪声、量化噪声等物理分量。
        """
        # Frame-wise 标定项保留用于兼容已发布权重和历史接口。
        if self.use_Frame:
            if 'fix' not in p:
                dh, dw = p['shape'][-2], p['shape'][-1]
                xx = np.random.randint(self.H - dh + 1)
                yy = np.random.randint(self.W - dw + 1)
                r = 0.05 #(torch.rand(1, device=p['device']) * r - r/2)
                if torch.rand(1) < 0.1:
                    r = r * 10
                ds = self.ds_k[xx:xx+dh, yy:yy+dw] * p['ISO'] * torch.randn(1, device=p['device']) * r
                ds += self.ds_b[xx:xx+dh, yy:yy+dw] * torch.randn(1, device=p['device']) * r
        # Band-wise Noise
        # if self.use_Band:
        #     if 'fix' not in p:
        #         raise NotImplementedError
        # else:
        N_row = N_col = 0
        # Pixel-wise Noise
        if self.use_Pixel:
            x_pre = p['x_pre'] if 'x_pre' in p else None
            x_fol = p['x_fol'] if 'x_fol' in p else None
            N_pixel = self.proxy_pixel(p, x_pre, x_fol)
            if 'fix' not in p:
                r = 0.05
                N_pixel = N_pixel * (1 + torch.randn(1, device=p['device']) * r)
        else:
            N_pixel = 0
        noise = N_row + N_col + N_pixel
        return noise