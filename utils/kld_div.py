from .utils import *
import torch
import torch.nn.functional as F

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

def kl_div_forward(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(p * np.log(p / q))


def kl_div_inverse(p, q):
    idx = ~(np.isnan(p) | np.isinf(p) | np.isnan(q) | np.isinf(q))
    p = p[idx]
    q = q[idx]
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    return np.sum(q * np.log(q / p))


def kl_div_sym(p, q):
    return (kl_div_forward(p, q) + kl_div_inverse(p, q)) / 2.0


def kl_div_3(p, q):
    kl_fwd = kl_div_forward(p, q)
    kl_inv = kl_div_inverse(p, q)
    kl_sym = (kl_inv + kl_fwd) / 2.0
    return kl_fwd, kl_inv, kl_sym


def kl_div_forward_data(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """ Forward KL divergence between two sets of data points p and q"""
    p, _ = get_histogram(p_data, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, left_edge, right_edge, n_bins)
    return kl_div_forward(p, q)


def kl_div_inverse_data(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """ Forward KL divergence between two sets of data points p and q"""
    p, _ = get_histogram(p_data, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, left_edge, right_edge, n_bins)
    return kl_div_inverse(p, q)


def kl_div_3_data(p_data, q_data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_fwd, kl_inv, kl_sym

def kl_div_norm(p_data, q_data, bin_edges=None, left_edge=0.0, right_edge=1.0, bl=512, wp=16383):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    l, r = min(p_data.min(), q_data.min()), max(p_data.max(), q_data.max())
    if bl is None:
        bl = 0
        n_bins = wp
        left_edge, right_edge = l, r
    else:
        if p_data.min() < 0:
            p_data += bl
            q_data += bl
        p_data = np.round(p_data)
        q_data = np.round(q_data)
        p_data = norm(p_data, 0, wp, clip=True)
        q_data = norm(q_data, 0, wp, clip=True)
        n_bins = wp 
    # print(n_bins)
    if bin_edges is None:
        data_range = right_edge - left_edge
        bin_width = data_range / n_bins
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    y_p, x_p = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    y_q, x_q = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    # plt.bar(x_p, y_p, bin_width, color='C0', alpha=0.7)
    # plt.bar(x_q, y_q, bin_width, color='C1', alpha=0.7)
    # plt.xlim(p_data.mean()-p_data.std()*3 , p_data.mean()+p_data.std()*3)
    idx = (y_p > 0) & (y_q > 0)
    p = y_p[idx]
    q = y_q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    # plt.show()
    results = {'kl_fwd':kl_fwd, 'kl_inv':kl_inv, 'kl_sym':kl_sym,
            'hist_p':(y_p, bin_edges*wp-bl), 'hist_q':(y_q, bin_edges*wp-bl)}
    return results

def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers