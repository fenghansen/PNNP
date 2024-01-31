from .utils import *

def read_wb_ccm(raw):
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    wb = wb.astype(np.float32)
    ccm = raw.color_matrix[:3, :3].astype(np.float32)
    if ccm[0,0] == 0:
        ccm = np.eye(3, dtype=np.float32)
    return wb, ccm

def get_ISO_ExposureTime(filepath):
    # 不限于RAW，RGB图片也适用
    raw_file = open(filepath, 'rb')
    exif_file = exifread.process_file(raw_file, details=False, strict=True)
    # 获取曝光时间
    if 'EXIF ExposureTime' in exif_file:
        exposure_str = exif_file['EXIF ExposureTime'].printable
    else:
        exposure_str = exif_file['Image ExposureTime'].printable
    if '/' in exposure_str:
        fenmu = float(exposure_str.split('/')[0])
        fenzi = float(exposure_str.split('/')[-1])
        exposure = fenmu / fenzi
    else:
        exposure = float(exposure_str)
    # 获取ISO
    if 'EXIF ISOSpeedRatings' in exif_file:
        ISO_str = exif_file['EXIF ISOSpeedRatings'].printable
    else:
        ISO_str = exif_file['Image ISOSpeedRatings'].printable
    if '/' in ISO_str:
        fenmu = float(ISO_str.split('/')[0])
        fenzi = float(ISO_str.split('/')[-1])
        ISO = fenmu / fenzi
    else:
        ISO = float(ISO_str)
    info = {'ISO':int(ISO), 'ExposureTime':exposure, 'name':filepath.split('/')[-1]}
    return info

def metainfo(rawpath):
    with open(rawpath, 'rb') as f:
        tags = exifread.process_file(f)
        _, suffix = os.path.splitext(os.path.basename(rawpath))

        if suffix == '.dng':
            expo = eval(str(tags['Image ExposureTime']))
            iso = eval(str(tags['Image ISOSpeedRatings']))
        else:
            expo = eval(str(tags['EXIF ExposureTime']))
            iso = eval(str(tags['EXIF ISOSpeedRatings']))

        # print('ISO: {}, ExposureTime: {}'.format(iso, expo))
    return iso, expo

# Yuzhi Wang's ISP
def bayer2rggb(bayer):
    H, W = bayer.shape
    return bayer.reshape(H//2, 2, W//2, 2).transpose(0, 2, 1, 3).reshape(H//2, W//2, 4)

def rggb2bayer(rggb):
    H, W, _ = rggb.shape
    return rggb.reshape(H, W, 2, 2).transpose(0, 2, 1, 3).reshape(H*2, W*2)

def bayer2rows(bayer):
    # 分行
    H, W = bayer.shape
    return np.stack((bayer[0:H:2], bayer[1:H:2]))

def bayer2gray(raw):
    # 相当于双线性插值的bayer2gray
    kernel = np.array([[1,2,1],[2,4,2],[1,2,1]], np.float32) / 16.
    gray = cv2.filter2D(raw, -1, kernel, borderType=cv2.BORDER_REFLECT)
    return gray

def rows2bayer(rows):
    c, H, W = rows.shape
    bayer = np.empty((H*2, W))
    bayer[0:H*2:2] = rows[0]
    bayer[1:H*2:2] = rows[1]
    return bayer

# Kaixuan Wei's ISP
def raw2bayer(raw, wp=1023, bl=64, norm=True, clip=False, bias=np.array([0,0,0,0])):
    raw = raw.astype(np.float32)
    H, W = raw.shape
    out = np.stack((raw[0:H:2, 0:W:2], #RGBG
                    raw[0:H:2, 1:W:2],
                    raw[1:H:2, 1:W:2],
                    raw[1:H:2, 0:W:2]), axis=0).astype(np.float32) 
    if norm:
        bl = bias + bl
        bl = bl.reshape(4, 1, 1) 
        out = (out - bl) / (wp - bl)
    if clip: out = np.clip(out, 0, 1)
    return out.astype(np.float32) 

def bayer2raw(packed_raw, wp=16383, bl=512):
    if torch.is_tensor(packed_raw):
        packed_raw = packed_raw.detach()
        packed_raw = packed_raw[0].cpu().float().numpy()
    packed_raw = np.clip(packed_raw, 0, 1)
    packed_raw = packed_raw * (wp - bl) + bl
    C, H, W = packed_raw.shape
    H *= 2
    W *= 2
    raw = np.empty((H, W), dtype=np.uint16)
    raw[0:H:2, 0:W:2] = packed_raw[0, :,:]
    raw[0:H:2, 1:W:2] = packed_raw[1, :,:]
    raw[1:H:2, 1:W:2] = packed_raw[2, :,:]
    raw[1:H:2, 0:W:2] = packed_raw[3, :,:]
    return raw

# Hansen Feng's ISP
def repair_bad_pixels(raw, bad_points, method='median'):
    fixed_raw = bayer2rggb(raw)
    for i in range(4):
        fixed_raw[:,:,i] = cv2.medianBlur(fixed_raw[:,:,i],3)
    fixed_raw = rggb2bayer(fixed_raw)
    # raw = (1-bpc_map) * raw + bpc_map * fixed_raw
    for p in bad_points:
        raw[p[0],p[1]] = fixed_raw[p[0],p[1]]
    return raw

def SimpleISP(raw, bl=512, wp=16383, wb=[2,1,1,2], gamma=2.2):
    # rggb2RGB (SimpleISP)
    raw = (raw.astype(np.float32) - bl) / (wp-bl)
    wb = np.array(wb)
    raw = raw * wb.reshape(1,1,-1)
    raw = raw.clip(0, 1)[:,:,(0,1,3)]
    raw = raw ** (1/gamma)
    return raw

def FastISP(img4c, wb=None, ccm=None, gamma=2.2):
    # rgbg2RGB (FastISP)
    c,h,w = img4c.shape
    H = h * 2
    W = w * 2
    raw = np.zeros((H,W), np.float32)
    red_gain = wb[0] if wb is not None else 2
    blue_gain = wb[2] if wb is not None else 2
    raw[0:H:2,0:W:2] = img4c[0] * red_gain # R
    raw[0:H:2,1:W:2] = img4c[1] # G1
    raw[1:H:2,1:W:2] = img4c[2] * blue_gain # B
    raw[1:H:2,0:W:2] = img4c[3] # G2
    raw = np.clip(raw, 0, 1)
    white_point = 16383
    raw = raw * white_point
    img = cv2.cvtColor(raw.astype(np.uint16), cv2.COLOR_BAYER_BG2RGB_EA) / white_point
    if ccm is None: # SonyCCM
        ccm = np.array( [[ 1.9712269,-0.6789218, -0.29230508],
                        [-0.29104823, 1.748401 , -0.45735288],
                        [ 0.02051281,-0.5380369,  1.5175241 ]])
    img = img[:, :, None, :]
    ccm = ccm[None, None, :, :]
    img = np.sum(img * ccm, axis=-1)
    img = np.clip(img, 0, 1) ** (1/gamma)
    return img

def raw2rgb_rawpy(packed_raw, wb=None, ccm=None):
    """Raw2RGB pipeline (rawpy postprocess version)"""
    if packed_raw.shape[-2] > 1500:
        raw = rawpy.imread('templet.dng')
        wp = 1023
        bl = 64
    else:
        raw = rawpy.imread('templet.ARW')
        wp = 16383
        bl = 512
    if wb is None:
        wb = np.array(raw.camera_whitebalance) 
        wb /= wb[1]
    wb = list(wb)
    if ccm is None:
        try:
            ccm = raw.rgb_camera_matrix[:3, :3]
        except:
            warnings.warn("You have no Wei Kaixuan's customized rawpy, you can't get right ccm of SonyA7S2...")
            ccm = raw.color_matrix[:3, :3]
    elif np.max(np.abs(ccm - np.identity(3))) == 0:
        ccm = np.array([[ 1.9712269,-0.6789218,-0.29230508],
                    [-0.29104823,1.748401,-0.45735288],
                    [ 0.02051281,-0.5380369,1.5175241 ]], dtype=np.float32)

    if len(packed_raw.shape) >= 3:
        raw.raw_image_visible[:] = bayer2raw(packed_raw, wp, bl)
    else: # 传进来的就是raw图
        raw.raw_image_visible[:] = packed_raw
        
    out = raw.postprocess(use_camera_wb=False, user_wb=wb, half_size=False, no_auto_bright=True, 
                        output_bps=8, bright=1, user_black=None, user_sat=None)
    return out