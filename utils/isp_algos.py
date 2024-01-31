from .isp_ops import *

# sigma是σ_read, gain是K
def VST(x, sigma, mu=0, gain=1.0, wp=1):
    # 无增益时，y = 2 * np.sqrt(x + 3.0 / 8.0 + sigma ** 2)
    y = x * wp
    y = gain * x + (gain ** 2) * 3.0 / 8.0 + sigma ** 2 - gain * mu
    y = np.sqrt(np.maximum(y, np.zeros_like(y)))
    y = y / wp
    return (2.0 / gain) * y

# sigma是σ_read, gain是K
def inverse_VST(x, sigma, gain=1.0, wp=1):
    x = x * wp
    y = (x / 2.0)**2 - 3.0/8.0 - sigma**2 / gain**2
    y_exact =  y * gain
    y_exact = y_exact / wp
    return y_exact

# 快速计算空域标准差
def stdfilt(img, k=5):
    # 计算均值图像和均值图像的平方图像
    img_blur = cv2.blur(img, (k, k))
    result_1 = img_blur ** 2
    # 计算图像的平方和平方后的均值
    img_2 = img ** 2
    result_2 = cv2.blur(img_2, (k, k))
    result = np.sqrt(np.maximum(result_2 - result_1, 0))
    return result

def Blur1D(data, c=0.5, log=True):
    l = len(data)
    if log:
        data = np.log2(data)
    temp = data.copy()
    for i in range(1, l-1):
        data[i] = temp[i] * c + (temp[i-1] + temp[i+1]) * (1-c)/2
    if log:
        data = 2 ** data 
    return data

def FastGuidedFilter(p,I,d=7,eps=1):
    p_lr = cv2.resize(p, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    I_lr = cv2.resize(I, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    mu_p = cv2.boxFilter(p_lr, -1, (d, d)) 
    mu_I = cv2.boxFilter(I_lr,-1, (d, d)) 
    
    II = cv2.boxFilter(np.multiply(I_lr,I_lr), -1, (d, d)) 
    Ip = cv2.boxFilter(np.multiply(I_lr,p_lr), -1, (d, d))
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.resize(cv2.boxFilter(a, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    mu_b = cv2.resize(cv2.boxFilter(b, -1, (d, d)), None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def GuidedFilter(p,I,d=7,eps=1):
    mu_p = cv2.boxFilter(p, -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    mu_I = cv2.boxFilter(I,-1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    
    II = cv2.boxFilter(np.multiply(I,I), -1, (d, d), borderType=cv2.BORDER_REPLICATE) 
    Ip = cv2.boxFilter(np.multiply(I,p), -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    var = II-np.multiply(mu_I,mu_I)
    cov = Ip-np.multiply(mu_I,mu_p)
    
    a = cov / (var + eps)
    
    b = mu_p - np.multiply(a,mu_I)
    mu_a = cv2.boxFilter(a, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    mu_b = cv2.boxFilter(b, -1, (d, d), borderType=cv2.BORDER_REPLICATE)
    
    dstImg = np.multiply(mu_a, I) + mu_b
    
    return dstImg

def row_denoise(path, iso, data=None):
    if data is None:
        raw = dataload(path)
    else:
        raw = data
    raw = bayer2rows(raw)
    raw_denoised = raw.copy()
    for i in range(len(raw)):
        rows = raw[i].mean(axis=1)
        rows2 = rows.reshape(1, -1)
        rows2 = cv2.bilateralFilter(rows2, 25, sigmaColor=10, sigmaSpace=1+iso/200, borderType=cv2.BORDER_REPLICATE)[0]
        row_diff = rows-rows2
        raw_denoised[i] = raw[i] - row_diff.reshape(-1, 1)
    raw = rows2bayer(raw)
    raw_denoised = rows2bayer(raw_denoised)
    return raw_denoised