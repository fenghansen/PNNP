"""Forward processing of raw data to sRGB images.
Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

import rawpy
import rawpy.enhance
import exifread
import numpy as np
import torch
import torch.nn as nn
import torch.distributions as tdist
from scipy import stats
from utils import *
import random
from .unprocess import random_gains

Dual_ISO_Cameras = ['SonyA7S2']
HALF_CLIP = 2

def data_aug(data, choice, bias=0, rot=False):
    if choice[0] == 1:
        data = np.flip(data, axis=2+bias)
    if choice[1] == 1:
        data = np.flip(data, axis=3+bias)
    return data

def inverse_VST_torch(x, noiseparam, iso_list, wp=1):
    x = x * wp
    b = len(iso_list)
    for i in range(b):
        iso = iso_list[i].item()
        gain = noiseparam[iso]['Kmax']
        sigma = noiseparam[iso]['sigGs']
        x[i] = (x[i] / 2.0)**2 - 3.0/8.0 - sigma**2 / gain**2
        x[i] =  x[i] * gain
    x = x / wp
    return x

def pack_raw_bayer(raw, wp=1023, clip=True):
    #pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    B = np.where(raw_pattern==2)
    G2 = np.where(raw_pattern==3)
    
    white_point = wp
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.stack((im[R[0][0]:H:2, R[1][0]:W:2], #RGBG
                    im[G1[0][0]:H:2, G1[1][0]:W:2],
                    im[B[0][0]:H:2, B[1][0]:W:2],
                    im[G2[0][0]:H:2, G2[1][0]:W:2]), axis=0).astype(np.float32)

    black_level = np.array(raw.black_level_per_channel)[:,None,None].astype(np.float32)

    out = (out - black_level) / (white_point - black_level)
    out = np.clip(out, 0.0, 1.0) if clip else out
    
    return out

def postprocess_bayer(rawpath, img4c, white_point=1023):
    if torch.is_tensor(img4c):
        img4c = img4c.detach()
        img4c = img4c[0].cpu().float().numpy()
    img4c = np.clip(img4c, 0, 1)

    #unpack 4 channels to Bayer image
    raw = rawpy.imread(rawpath)
    raw_pattern = raw.raw_pattern
    R = np.where(raw_pattern==0)
    G1 = np.where(raw_pattern==1)
    G2 = np.where(raw_pattern==3)
    B = np.where(raw_pattern==2)
    
    black_level = np.array(raw.black_level_per_channel)[:,None,None]

    img4c = img4c * (white_point - black_level) + black_level
    
    img_shape = raw.raw_image_visible.shape
    H = img_shape[0]
    W = img_shape[1]

    raw.raw_image_visible[R[0][0]:H:2, R[1][0]:W:2] = img4c[0, :,:]
    raw.raw_image_visible[G1[0][0]:H:2,G1[1][0]:W:2] = img4c[1, :,:]
    raw.raw_image_visible[B[0][0]:H:2,B[1][0]:W:2] = img4c[2, :,:]
    raw.raw_image_visible[G2[0][0]:H:2,G2[1][0]:W:2] = img4c[3, :,:]
    
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1,1,1,1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    # out = raw.postprocess(use_camera_wb=False, user_wb=[1.96875, 1, 1.444, 1], half_size=True, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)    
    out = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8, bright=1, user_black=None, user_sat=None)
    return out

def postprocess_bayer_v2(rawpath, img4c):    
    with rawpy.imread(rawpath) as raw:
        out_srgb = raw2rgb_postprocess(img4c.detach(), raw)        
    
    return out_srgb

def apply_gains(bayer_images, wbs):
    """Applies white balance to a batch of Bayer images."""
    N, C, _, _ = bayer_images.shape
    wbs = wbs.repeat((N,1)).view(N, C, 1, 1)
    outs = bayer_images * wbs
    return outs


def apply_ccms(images, ccms):
    """Applies color correction matrices."""
    images = images.permute(
        0, 2, 3, 1)  # Permute the image tensor to BxHxWxC format from BxCxHxW format
    images = images[:, :, :, None, :]
    ccms = ccms[:, None, None, :, :]
    outs = torch.sum(images * ccms, dim=-1)
    # Re-Permute the tensor back to BxCxHxW format
    outs = outs.permute(0, 3, 1, 2)
    return outs


def gamma_compression(images, gamma=2.2):
    """Converts from linear to gamma space."""
    outs = torch.clamp(images, min=1e-8) ** (1 / gamma)
    # outs = (1 + gamma[0]) * np.power(images, 1.0/gamma[1]) - gamma[0] + gamma[2]*images
    outs = torch.clamp((outs*255).int(), min=0, max=255).float() / 255
    return outs


def raw2LRGB(bayer_images): 
    """RGBG -> linear RGB"""
    lin_rgb = torch.stack([
        bayer_images[:,0,...], 
        torch.mean(bayer_images[:, [1,3], ...], dim=1), 
        bayer_images[:,2,...]], dim=1)

    return lin_rgb


def process(bayer_images, wbs, cam2rgbs, gamma=2.2):
    """Processes a batch of Bayer RGBG images into sRGB images."""
    # White balance.
    bayer_images = apply_gains(bayer_images, wbs)
    # Binning
    bayer_images = torch.clamp(bayer_images, min=0.0, max=1.0)
    images = raw2LRGB(bayer_images)
    # Color correction.
    images = apply_ccms(images, cam2rgbs)
    # Gamma compression.
    images = torch.clamp(images, min=0.0, max=1.0)
    images = gamma_compression(images, gamma)
    
    return images


def raw2rgb(packed_raw, raw):
    """Raw2RGB pipeline (preprocess version)"""
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.color_matrix[:3, :3]
    if cam2rgb[0,0] == 0:
        cam2rgb = np.eye(3, dtype=np.float32)

    if isinstance(packed_raw, np.ndarray):
        packed_raw = torch.from_numpy(packed_raw).float()

    wb = torch.from_numpy(wb).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb).float().to(packed_raw.device)
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2)[0, ...].numpy()
    
    return out


def raw2rgb_v2(packed_raw, wb, ccm):
    if torch.is_tensor(packed_raw):
        packed_raw = packed_raw.detach().cpu().float()
    else:
        packed_raw = torch.from_numpy(packed_raw).float()
    wb = torch.from_numpy(wb).float()
    cam2rgb = torch.from_numpy(ccm).float()
    out = process(packed_raw[None], wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2)[0, ...].numpy()
    return out.transpose(1,2,0)


def raw2rgb_postprocess(packed_raw, raw):
    """Raw2RGB pipeline (postprocess version)"""
    assert packed_raw.ndimension() == 4
    wb = np.array(raw.camera_whitebalance) 
    wb /= wb[1]
    cam2rgb = raw.color_matrix[:3, :3]
    if cam2rgb[0,0] == 0:
        cam2rgb = np.eye(3, dtype=np.float32)

    wb = torch.from_numpy(wb[None]).float().to(packed_raw.device)
    cam2rgb = torch.from_numpy(cam2rgb[None]).float().to(packed_raw.device)
    out = process(packed_raw, wbs=wb, cam2rgbs=cam2rgb, gamma=2.2)
    # out = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
    return out

# def get_specific_noise_params(camera_type=None, iso='100'):
#     cam_noisy_params = {}
#     cam_noisy_params['IMX686'] = {
#         '100':{'K':0.1366021, 'sigGs':0.6926457, 'sigGssig':0.002096},
#         '6400':{'K':8.7425333, 'sigGs':14.303619546153575, 'sigGssig':0.0696716845864088},

#     }
#     if camera_type in cam_noisy_params:
#         return cam_noisy_params[camera_type][iso]
#     else:
#         log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use NikonD850's parameters to test.''')
#         return cam_noisy_params['IMX686']

def get_camera_noisy_params(camera_type=None):
    cam_noisy_params = {}
    cam_noisy_params['NikonD850'] = {
        'Kmin':1.2, 'Kmax':2.4828, 'lam':-0.26, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'sigTLk':0.906, 'sigTLb':-0.6754,   'sigTLsig':0.035165,
        'sigRk':0.8322,  'sigRb':-2.3326,   'sigRsig':0.301333,
        'sigGsk':0.8322, 'sigGsb':-0.1754,  'sigGssig':0.035165,
    }
    cam_noisy_params['IMX686'] = { # ISO-640~6400
        'Kmin':-0.19118, 'Kmax':2.16820, 'lam':0.102, 'q':1/(2**10), 'wp':1023, 'bl':64,
        'sigTLk':0.85187, 'sigTLb':0.07991,   'sigTLsig':0.02921,
        'sigRk':0.87611,  'sigRb':-2.11455,   'sigRsig':0.03274,
        'sigGsk':0.85187, 'sigGsb':0.67991,   'sigGssig':0.02921,
    }
    cam_noisy_params['SonyA7S2_lowISO'] = {
        'Kmin':-1.67214, 'Kmax':0.42228, 'lam':-0.026, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'sigRk':0.78782,  'sigRb':-0.34227,  'sigRsig':0.02832,
        'sigTLk':0.74043, 'sigTLb':0.86182, 'sigTLsig':0.00712,
        'sigGsk':0.82966, 'sigGsb':1.49343, 'sigGssig':0.00359,
        'sigReadk':0.82879, 'sigReadb':1.50601, 'sigReadsig':0.00362,
        'uReadk':0.01472, 'uReadb':0.01129, 'uReadsig':0.00034,
    }
    cam_noisy_params['SonyA7S2_highISO'] = {
        'Kmin':0.64567, 'Kmax':2.51606, 'lam':-0.025, 'q':1/(2**14), 'wp':16383, 'bl':512,
        'sigRk':0.62945,  'sigRb':-1.51040,  'sigRsig':0.02609,
        'sigTLk':0.74901, 'sigTLb':-0.12348, 'sigTLsig':0.00638,
        'sigGsk':0.82878, 'sigGsb':0.44162, 'sigGssig':0.00153,
        'sigReadk':0.82645, 'sigReadb':0.45061, 'sigReadsig':0.00156,
        'uReadk':0.00385, 'uReadb':0.00674, 'uReadsig':0.00039,
    }
    cam_noisy_params['CRVD'] = {
        'Kmin':1.31339, 'Kmax':3.95448, 'lam':0.015, 'q':1/(2**12), 'wp':4095, 'bl':240,
        'sigRk':0.93368,  'sigRb':-2.19692,  'sigRsig':0.02473,
        'sigGsk':0.95387, 'sigGsb':0.01552, 'sigGssig':0.00855,
        'sigTLk':0.95495, 'sigTLb':0.01618, 'sigTLsig':0.00790
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type]
    else:
        log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}". Now we use NikonD850's parameters to test.''')
        return cam_noisy_params['NikonD850']

def get_specific_noise_params(camera_type=None, iso='100'):
    iso = str(iso)
    cam_noisy_params = {}
    cam_noisy_params['SonyA7S2'] = {
        '50': {'Kmax': 0.047815, 'lam': 0.1474653, 'sigGs': 1.0164667, 'sigGssig': 0.005272454, 'sigTL': 0.70727646, 'sigTLsig': 0.004360543, 'sigR': 0.13997398, 'sigRsig': 0.0064381803, 'bias': 0, 'biassig': 0.010093017, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '64': {'Kmax': 0.0612032, 'lam': 0.13243394, 'sigGs': 1.0509665, 'sigGssig': 0.008081373, 'sigTL': 0.71535635, 'sigTLsig': 0.0056863446, 'sigR': 0.14346549, 'sigRsig': 0.006400559, 'bias': 0, 'biassig': 0.008690166, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '80': {'Kmax': 0.076504, 'lam': 0.1121489, 'sigGs': 1.180899, 'sigGssig': 0.011333668, 'sigTL': 0.7799473, 'sigTLsig': 0.009347968, 'sigR': 0.19540153, 'sigRsig': 0.008197397, 'bias': 0, 'biassig': 0.0107246125, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512},
        '100': {'Kmax': 0.09563, 'lam': 0.14875287, 'sigGs': 1.0067395, 'sigGssig': 0.0033682834, 'sigTL': 0.70181876, 'sigTLsig': 0.0037532174, 'sigR': 0.1391465, 'sigRsig': 0.006530218, 'bias': 0, 'biassig': 0.007235429, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '125': {'Kmax': 0.1195375, 'lam': 0.12904578, 'sigGs': 1.0279676, 'sigGssig': 0.007364685, 'sigTL': 0.6961967, 'sigTLsig': 0.0048687346, 'sigR': 0.14485553, 'sigRsig': 0.006731584, 'bias': 0, 'biassig': 0.008026363, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '160': {'Kmax': 0.153008, 'lam': 0.094135, 'sigGs': 1.1293099, 'sigGssig': 0.008340453, 'sigTL': 0.7258587, 'sigTLsig': 0.008032158, 'sigR': 0.19755602, 'sigRsig': 0.0082754735, 'bias': 0, 'biassig': 0.0101351, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '200': {'Kmax': 0.19126, 'lam': 0.07902429, 'sigGs': 1.2926387, 'sigGssig': 0.012171176, 'sigTL': 0.8117464, 'sigTLsig': 0.010250768, 'sigR': 0.22815849, 'sigRsig': 0.010726711, 'bias': 0, 'biassig': 0.011413908, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '250': {'Kmax': 0.239075, 'lam': 0.051688068, 'sigGs': 1.4345995, 'sigGssig': 0.01606571, 'sigTL': 0.8630922, 'sigTLsig': 0.013844714, 'sigR': 0.26271912, 'sigRsig': 0.0130637, 'bias': 0, 'biassig': 0.013569083, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '320': {'Kmax': 0.306016, 'lam': 0.040700804, 'sigGs': 1.7481371, 'sigGssig': 0.019626873, 'sigTL': 1.0334468, 'sigTLsig': 0.017629284, 'sigR': 0.3097104, 'sigRsig': 0.016202712, 'bias': 0, 'biassig': 0.017825918, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '400': {'Kmax': 0.38252, 'lam': 0.0222538, 'sigGs': 2.0595572, 'sigGssig': 0.024872316, 'sigTL': 1.1816813, 'sigTLsig': 0.02505812, 'sigR': 0.36209714, 'sigRsig': 0.01994737, 'bias': 0, 'biassig': 0.021005306, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '500': {'Kmax': 0.47815, 'lam': -0.0031342343, 'sigGs': 2.3956928, 'sigGssig': 0.030144656, 'sigTL': 1.31772, 'sigTLsig': 0.028629242, 'sigR': 0.42528257, 'sigRsig': 0.025104137, 'bias': 0, 'biassig': 0.02981831, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '640': {'Kmax': 0.612032, 'lam': 0.002566592, 'sigGs': 2.9662898, 'sigGssig': 0.045661453, 'sigTL': 1.6474211, 'sigTLsig': 0.04671843, 'sigR': 0.48839623, 'sigRsig': 0.031589635, 'bias': 0, 'biassig': 0.10000693, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '800': {'Kmax': 0.76504, 'lam': -0.008199721, 'sigGs': 3.5475867, 'sigGssig': 0.052318197, 'sigTL': 1.9346539, 'sigTLsig': 0.046128694, 'sigR': 0.5723769, 'sigRsig': 0.037824076, 'bias': 0, 'biassig': 0.025339302, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '1000': {'Kmax': 0.9563, 'lam': -0.021061005, 'sigGs': 4.2727833, 'sigGssig': 0.06972333, 'sigTL': 2.2795107, 'sigTLsig': 0.059203167, 'sigR': 0.6845563, 'sigRsig': 0.04879781, 'bias': 0, 'biassig': 0.027911892, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '1250': {'Kmax': 1.195375, 'lam': -0.032423194, 'sigGs': 5.177596, 'sigGssig': 0.092677385, 'sigTL': 2.708437, 'sigTLsig': 0.07622563, 'sigR': 0.8177013, 'sigRsig': 0.06162229, 'bias': 0, 'biassig': 0.03293372, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '1600': {'Kmax': 1.53008, 'lam': -0.0441045, 'sigGs': 6.29925, 'sigGssig': 0.1153261, 'sigTL': 3.2283993, 'sigTLsig': 0.09118158, 'sigR': 0.988786, 'sigRsig': 0.078567736, 'bias': 0, 'biassig': 0.03877672, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '2000': {'Kmax': 1.9126, 'lam': -0.012963797, 'sigGs': 2.653871, 'sigGssig': 0.015890995, 'sigTL': 1.4356787, 'sigTLsig': 0.02178686, 'sigR': 0.33124214, 'sigRsig': 0.018801652, 'bias': 0, 'biassig': 0.01570677, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '2500': {'Kmax': 2.39075, 'lam': -0.027097283, 'sigGs': 3.200225, 'sigGssig': 0.019307792, 'sigTL': 1.6897862, 'sigTLsig': 0.025873765, 'sigR': 0.38264316, 'sigRsig': 0.023769397, 'bias': 0, 'biassig': 0.018728448, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '3200': {'Kmax': 3.06016, 'lam': -0.034863412, 'sigGs': 3.9193838, 'sigGssig': 0.02649232, 'sigTL': 2.0417721, 'sigTLsig': 0.032873377, 'sigR': 0.44543457, 'sigRsig': 0.030114045, 'bias': 0, 'biassig': 0.021355819, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '4000': {'Kmax': 3.8252, 'lam': -0.043700505, 'sigGs': 4.8015847, 'sigGssig': 0.03781628, 'sigTL': 2.4629273, 'sigTLsig': 0.042401053, 'sigR': 0.52347374, 'sigRsig': 0.03929801, 'bias': 0, 'biassig': 0.026152484, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '5000': {'Kmax': 4.7815, 'lam': -0.053150143, 'sigGs': 5.8995814, 'sigGssig': 0.0625814, 'sigTL': 2.9761007, 'sigTLsig': 0.061326735, 'sigR': 0.6190265, 'sigRsig': 0.05335372, 'bias': 0, 'biassig': 0.058574405, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '6400': {'Kmax': 6.12032, 'lam': -0.07517104, 'sigGs': 7.1163535, 'sigGssig': 0.08435366, 'sigTL': 3.4502964, 'sigTLsig': 0.08226275, 'sigR': 0.7218788, 'sigRsig': 0.0642334, 'bias': 0, 'biassig': 0.059074216, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '8000': {'Kmax': 7.6504, 'lam': -0.08208357, 'sigGs': 8.916516, 'sigGssig': 0.12763213, 'sigTL': 4.269624, 'sigTLsig': 0.13381928, 'sigR': 0.87760293, 'sigRsig': 0.07389065, 'bias': 0, 'biassig': 0.084842026, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '10000': {'Kmax': 9.563, 'lam': -0.073289566, 'sigGs': 11.291476, 'sigGssig': 0.1639773, 'sigTL': 5.495318, 'sigTLsig': 0.16279395, 'sigR': 1.0522343, 'sigRsig': 0.094359785, 'bias': 0, 'biassig': 0.107438326, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '12800': {'Kmax': 12.24064, 'lam': -0.06495205, 'sigGs': 14.245901, 'sigGssig': 0.17283991, 'sigTL': 7.038261, 'sigTLsig': 0.18822834, 'sigR': 1.2749791, 'sigRsig': 0.120479785, 'bias': 0, 'biassig': 0.0944684, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '16000': {'Kmax': 15.3008, 'lam': -0.060692135, 'sigGs': 17.833515, 'sigGssig': 0.19809262, 'sigTL': 8.877547, 'sigTLsig': 0.23338738, 'sigR': 1.5559287, 'sigRsig': 0.15791349, 'bias': 0, 'biassig': 0.09725099, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '20000': {'Kmax': 19.126, 'lam': -0.060213074, 'sigGs': 22.084776, 'sigGssig': 0.21820943, 'sigTL': 11.002351, 'sigTLsig': 0.28806436, 'sigR': 1.8810822, 'sigRsig': 0.18937257, 'bias': 0, 'biassig': 0.4984733, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}, 
        '25600': {'Kmax': 24.48128, 'lam': -0.09089118, 'sigGs': 25.853043, 'sigGssig': 0.35371417, 'sigTL': 12.175712, 'sigTLsig': 0.4215717, 'sigR': 2.2760193, 'sigRsig': 0.2609267, 'bias': 0, 'biassig': 0.37568903, 'q': 6.103515625e-05, 'wp': 16383, 'bl': 512}
        }
    cam_noisy_params['IMX686'] = {
        '100':{
            'Kmax':0.083805, 'sigGs':0.6926457, 'sigGssig':0.002096,
            'sigTL':0.67998, 'lam':0.015, 'sigR':0.23668,
            'q':1/(2**10), 'wp':1023, 'bl':64, 
            'bias':np.array([0,0,0,0])
        },
        '6400':{
            'Kmax':8.74253, 'sigGs':14.30362, 'sigGssig':0.06967,
            'sigTL':12.8901, 'lam':0.015, 'sigR':0,
            'q':1/(2**10), 'wp':1023, 'bl':64, 
            'bias':np.array([-0.08113494,-0.04906388,-0.9408157,-1.2048522])
        }
    }
    if camera_type in cam_noisy_params:
        return cam_noisy_params[camera_type][iso]
    else:
        # log(f'''Warning: we have not test the noisy parameters of camera "{camera_type}".''')
        return None

# 取最大对应相机的最大ISO生成噪声参数
def sample_params_max(camera_type='NikonD850', ratio=None, iso=None):
    # 获取已经测算好的相机噪声参数
    params = None
    if iso is not None:
        params = get_specific_noise_params(camera_type=camera_type, iso=iso)
    if params is None:
        if camera_type in Dual_ISO_Cameras:
            choice = np.random.randint(2)
            camera_type += '_lowISO' if choice<1 else '_highISO'
        params = get_camera_noisy_params(camera_type=camera_type)
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        bias = 0
        log_K = params['Kmax'] + np.random.uniform(low=-0.01, high=+0.01) # 增加一些扰动，以防测的不准
        K = np.exp(log_K)
        mu_TL = params['sigTLk']*log_K + params['sigTLb']
        mu_R = params['sigRk']*log_K + params['sigRb']
        mu_Gs = params['sigGsk']*log_K + params['sigGsb'] if 'sigGsk' in params else 2**(-14)
        # 去掉log
        sigTL = np.exp(mu_TL)
        sigR = np.exp(mu_R)
        sigGs = np.exp(np.random.normal(loc=mu_Gs, scale=params['sigGssig']) if 'sigGssig' in params else mu_Gs)
    else:
        K = params['Kmax'] * (1 + np.random.uniform(low=-0.01, high=+0.01)) # 增加一些扰动，以防测的不准
        sigGs = np.random.normal(loc=params['sigGs'], scale=params['sigGssig']) if 'sigGssig' in params else params['sigGs']
        sigTL = np.random.normal(loc=params['sigTL'], scale=params['sigTLsig']) if 'sigTLsig' in params else params['sigTL']
        sigR = np.random.normal(loc=params['sigR'], scale=params['sigRsig']) if 'sigRsig' in params else params['sigR']
        bias = params['bias']
    wp = params['wp']
    bl = params['bl']
    lam = params['lam']
    q = params['q']

    if ratio is None:
        if 'SonyA7S2' in camera_type:
            ratio = np.random.uniform(low=100, high=300)
        else:
            log_ratio = np.random.uniform(low=0, high=2.08)
            ratio = np.exp(log_ratio)

    return {'K':K, 'sigTL':sigTL, 'sigR':sigR, 'sigGs':sigGs, 'bias':bias,
            'lam':lam, 'q':q, 'ratio':ratio, 'wp':wp, 'bl':bl}

# 噪声参数采样
def sample_params(camera_type='NikonD850', ln_ratio=False):
    choice = 1
    Dual_ISO_Cameras = ['SonyA7S2']
    if camera_type in Dual_ISO_Cameras:
        choice = np.random.randint(2)
        camera_type += '_lowISO' if choice<1 else '_highISO'

    # 获取已经测算好的相机噪声参数
    params = get_camera_noisy_params(camera_type=camera_type)
    wp = params['wp']
    bl = params['bl']
    lam = params['lam']
    q = params['q']
    Point_ISO_Cameras = ['CRVD', 'BM3D']
    if camera_type in Point_ISO_Cameras:
        if camera_type == 'CRVD':
            iso_list = [1600,3200,6400,12800,25600]
            a_list = np.array([3.513262,6.955588,13.486051,26.585953,52.032536])
            b_list = np.array([11.917691,38.117816,130.818508,484.539790,1819.818657])
            bias_points = np.array([-1.12660, -1.69546, -3.25935, -6.68111, -12.66876])
            K_points = np.log(a_list)
            Gs_points = np.log(np.sqrt(b_list))
        choice = np.random.randint(5)
        # 取点噪声参数
        log_K = K_points[choice]
        K = a_list[choice]
        mu_TL = params['sigTLk']*log_K + params['sigTLb'] if 'sigTLk' in params else 0
        mu_R = params['sigRk']*log_K + params['sigRb'] if 'sigRk' in params else 0
        mu_Gs = Gs_points[choice]
        bias = bias_points[choice]
    else:
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        bias = 0
        log_K = np.random.uniform(low=params['Kmin'], high=params['Kmax'])
        K = np.exp(log_K)
        mu_TL = params['sigTLk']*log_K + params['sigTLb'] if 'sigTLk' in params else q
        mu_R = params['sigRk']*log_K + params['sigRb'] if 'sigRk' in params else q
        mu_Gs = params['sigGsk']*log_K + params['sigGsb'] if 'sigGsk' in params else q
        mu_bias = params['uReadk']*log_K + params['uReadb']

    log_sigTL = np.random.normal(loc=mu_TL, scale=params['sigTLsig']) if 'sigTLk' in params else 0
    log_sigR = np.random.normal(loc=mu_R, scale=params['sigRsig']) if 'sigRk' in params else 0
    log_sigGs = np.random.normal(loc=mu_Gs, scale=params['sigGssig']) if 'sigGsk' in params else q
    log_bias = np.random.normal(loc=mu_bias, scale=params['uReadsig']) if 'uReadk' in params else 0
    # 去掉log
    sigTL = np.exp(log_sigTL)
    sigR = np.exp(log_sigR)
    sigGs = np.exp(log_sigGs)
    bias = np.exp(log_bias)
    # 模拟曝光衰减的系数, ln_ratio模式会照顾弱噪声场景，更有通用性
    if ln_ratio:
        high = 1 if 'CRVD' in camera_type else 5
        log_ratio = np.random.uniform(low=-0.01, high=high)
        ratio = np.exp(log_ratio) # np.random.uniform(low=1, high=200) if choice else np.exp(log_ratio)
    else:
        ratio = np.random.uniform(low=100, high=300)
        
    return {'K':K, 'sigTL':sigTL, 'sigR':sigR, 'sigGs':sigGs, 'bias':bias,
            'lam':lam, 'q':q, 'ratio':ratio, 'wp':wp, 'bl':bl}


def get_aug_param_torch(data, b=8, command='augv5', numpy=False, camera_type='SonyA7S2'):
    aug_r, aug_g, aug_b = torch.zeros(b), torch.zeros(b), torch.zeros(b)
    r = np.random.randint(2) * 0.25 + 0.25
    u = r
    if np.random.randint(4):
        if 'augv5' in command:
            # v5，依照相机白平衡经验参数增强，激进派作风
            rgb_gain, red_gain, blue_gain = random_gains(camera_type)
            rgb_gain = 1 / rgb_gain
            rg = data["wb"][:, 0] / red_gain[0]
            bg = data["wb"][:, 2] / blue_gain[0]
            aug_g = torch.rand(b) * r + rgb_gain[0] - 0.9
            aug_r = torch.rand(b) * r + rg * (1+aug_g) - 1.1
            aug_b = torch.rand(b) * r + bg * (1+aug_g) - 1.1
        elif 'augv2' in command:
            # v2，相信原数据颜色的准确性，保守派作风
            aug_g = torch.clamp(torch.randn(b) * r, 0, 4*u)
            aug_r = torch.clamp((1+torch.randn(b)*r) * (1+aug_g) - 1, 0, 4*u)
            aug_b = torch.clamp((1+torch.randn(b)*r) * (1+aug_g) - 1, 0, 4*u)

    # 保证非负
    daug, _ = torch.min(torch.stack((aug_r, aug_g, aug_b)), dim=0)
    daug[daug>0] = 0
    aug_r = (1+aug_r) / (1+daug) - 1
    aug_g = (1+aug_g) / (1+daug) - 1
    aug_b = (1+aug_b) / (1+daug) - 1
    if numpy:
        aug_r = np.squeeze(aug_r.numpy())
        aug_g = np.squeeze(aug_g.numpy())
        aug_b = np.squeeze(aug_b.numpy())
    return aug_r, aug_g, aug_b

def raw_wb_aug(noisy, gt, aug_wb=None, camera_type='SonyA7S2', ratio=1, ori=True, iso=None):
    # [c, h, w]
    p = get_specific_noise_params(camera_type=camera_type, iso=iso)
    if p is None:
        assert camera_type == 'SonyA7S2'
        camera_type += '_lowISO' if iso<=1600 else '_highISO'
        p = get_camera_noisy_params(camera_type=camera_type)
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        p['K'] = 0.0009546 * iso * (1 + np.random.uniform(low=-0.01, high=+0.01)) - 0.00193
        log_K = np.log(p['K'])
        mu_Gs = p['sigGsk']*log_K + p['sigGsb']
        p['sigGs'] = np.exp(np.random.normal(loc=mu_Gs, scale=p['sigGssig']))
    else:
        p['K'] = p['Kmax'] * (1 + np.random.uniform(low=-0.01, high=+0.01)) # 增加一些扰动，以防测的不准
        p['sigGs'] = np.random.normal(loc=p['sigGs'], scale=p['sigGssig']) if 'sigGssig' in p else p['sigGs']
    
    if aug_wb is not None:
        # 默认pattern为RGGB！(通道rgbg排列)
        gt = gt * (p['wp'] - p['bl']) / ratio
        noisy = noisy * (p['wp'] - p['bl'])
        # 补噪声
        daug = -np.minimum(np.min(aug_wb), 0)
        # aug_wb = aug_wb + daug
        if daug == 0:
            # 只有增益的话很好处理，叠加泊松分布就行
            dy = gt * aug_wb.reshape(-1,1,1)    # 我考虑过这里dy和dn要不要量化一下，感觉不量化对多样性更友好
            dn = np.random.poisson(dy/p['K']).astype(np.float32) * p['K']
        else:
            # BiSNA，折中方案，不好讲故事，弃疗了
            raise NotImplementedError
            # 存在减益的话就很麻烦，需要考虑read noise并且补齐分布
            scale = 1 - daug
            # 要保证dyn是非负的
            aug_wb_new = aug_wb + daug
            dy = gt * aug_wb.reshape(-1,1,1)
            dyn = gt * aug_wb_new.reshape(-1,1,1)
            # 先通过缩放减小噪声图
            noisy *= scale
            # 补齐单个照片的读噪声
            dn_read = np.random.randn(*gt.shape).astype(np.float32) * p['sigGs'] * np.sqrt(1-scale**2)
            # 补齐由于除法导致的分布变化，恢复shot noise应有的分布（不完全相等）
            scale_sigma = scale - scale**2
            dn_shot = np.random.poisson(scale_sigma * gt/p['K']).astype(np.float32)*p['K'] - gt * scale_sigma
            # 叠加泊松分布
            dn_aug = np.random.poisson(dyn/p['K']).astype(np.float32) * p['K']
            dn = dn_read + dn_shot + dn_aug

        gt = np.clip((gt + dy)*ratio, 0, (p['wp'] - p['bl']))
        noisy = np.clip(noisy + dn, -p['bl'], (p['wp'] - p['bl']))
        gt /= (p['wp'] - p['bl'])
        noisy /= (p['wp'] - p['bl'])

    if ori is False:
        noisy *= ratio
    
    return noisy.astype(np.float32), gt.astype(np.float32)

def raw_wb_aug_torch(noisy, gt, aug_wb=None, camera_type='IMX686', ratio=1, ori=True, iso=None, ratiofix=False):
    p = get_specific_noise_params(camera_type=camera_type, iso=iso)
    if p is None:
        assert camera_type == 'SonyA7S2'
        camera_type += '_lowISO' if iso<=1600 else '_highISO'
        p = get_camera_noisy_params(camera_type=camera_type)
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        p['K'] = 0.0009546 * iso * (1 + np.random.uniform(low=-0.01, high=+0.01)) - 0.00193
        log_K = np.log(p['K'])
        mu_Gs = p['sigGsk']*log_K + p['sigGsb']
        p['sigGs'] = np.exp(np.random.normal(loc=mu_Gs, scale=p['sigGssig']))
    else:
        p['K'] = p['Kmax'] * (1 + np.random.uniform(low=-0.01, high=+0.01)) # 增加一些扰动，以防测的不准
        p['sigGs'] = np.random.normal(loc=p['sigGs'], scale=p['sigGssig']) if 'sigGssig' in p else p['sigGs']
    
    if aug_wb is not None:
        # 默认pattern为RGGB！(通道rgbg排列)
        gt = gt * (p['wp'] - p['bl']) / ratio
        noisy = noisy * (p['wp'] - p['bl'])
        # 补噪声
        daug = -np.minimum(np.min(aug_wb), 0)
        daug = torch.from_numpy(np.array(daug)).to(noisy.device)
        aug_wb = torch.from_numpy(aug_wb).to(noisy.device)
        dy = gt * aug_wb.reshape(-1,1,1)    # 我考虑过这里dy和dn要不要量化一下，感觉不量化对多样性更友好
        if daug == 0:
            # 只有增益的话很好处理，叠加泊松分布就行
            dn = tdist.Poisson(dy/p['K']).sample() * p['K']
        else:
            # BiSNA，折中方案，不好讲故事，弃疗了
            warnings.warn('You are using BiSNA!!!')
            raise NotImplementedError
            # 存在减益的话就很麻烦，需要考虑read noise并且补齐分布
            scale = 1 - daug
            # 要保证dyn是非负的
            aug_wb_new = aug_wb + daug
            dyn = gt * aug_wb_new.reshape(-1,1,1)
            # 先通过缩放减小噪声图
            noisy *= scale
            # 补齐单个照片的读噪声
            dn_read = tdist.Normal(0, p['sigGs']).sample() * torch.sqrt(1-scale**2)
            # 补齐由于除法导致的分布变化，恢复shot noise应有的分布（不完全相等）
            scale_sigma = scale - scale**2
            dn_shot = tdist.Poisson(scale_sigma * gt/p['K']).sample() *p['K'] - gt * scale_sigma
            # 叠加泊松分布
            dn_aug = tdist.Poisson(dyn/p['K']).sample() * p['K']
            dn = dn_read + dn_shot + dn_aug
        if ratiofix:
            ratio = ratio / (1 + daug)
        gt = torch.clamp((gt + dy)*ratio, 0, (p['wp'] - p['bl']))
        noisy = torch.clamp(noisy + dn, -p['bl'], (p['wp'] - p['bl']))
        gt /= (p['wp'] - p['bl'])
        noisy /= (p['wp'] - p['bl'])

    if ori is False:
        noisy *= ratio
    
    return noisy, gt

def SNA_torch(gt, aug_wb, camera_type='IMX686', ratio=1, black_lr=False, ori=True, iso=None):
    p = get_specific_noise_params(camera_type=camera_type, iso=iso)
    if p is None:
        assert camera_type == 'SonyA7S2'
        camera_type += '_lowISO' if iso<=1600 else '_highISO'
        p = get_camera_noisy_params(camera_type=camera_type)
        # 根据最小二乘法得到的噪声参数回归模型采样噪声参数
        p['K'] = 0.0009546 * iso * (1 + np.random.uniform(low=-0.01, high=+0.01)) - 0.00193
    else:
        p['K'] = p['Kmax'] * (1 + np.random.uniform(low=-0.01, high=+0.01)) # 增加一些扰动，以防测的不准
    
    # 默认pattern为RGGB！(通道rgbg排列)
    gt = gt * (p['wp'] - p['bl']) / ratio
    # 补噪声
    aug_wb = torch.from_numpy(aug_wb).to(gt.device)
    dy = gt * aug_wb.reshape(-1,1,1)    # 我考虑过这里dy和dn要不要量化一下，感觉不量化对多样性更友好
    # 只有增益的话很好处理，叠加泊松分布就行
    dn = tdist.Poisson(dy/p['K']).sample() * p['K']
    # 贴黑图的，所以抛去gt中多算的一份泊松分布
    if black_lr: dy = dy - gt
    dy = dy * ratio / (p['wp'] - p['bl'])
    dn = dn / (p['wp'] - p['bl'])

    if ori is False:
        dn *= ratio
    
    return dn, dy

# @ fn_timer
def generate_noisy_obs(y, camera_type=None, wp=16383, noise_code='p', param=None, MultiFrameMean=1, ori=False, clip=False):
    p = param
    y = y * (p['wp'] - p['bl'])
    # p['ratio'] = 1/p['ratio'] # 临时行为，为了快速实现MFM
    y = y / p['ratio']
    MFM = MultiFrameMean ** 0.5

    use_R = True if 'r' in noise_code.lower() else False
    use_Q = True if 'q' in noise_code.lower() else False
    use_TL = True if 'g' in noise_code.lower() else False
    use_P = True if 'p' in noise_code.lower() else False
    use_D = True if 'd' in noise_code.lower() else False
    use_black = True if 'b' in noise_code.lower() else False
    
    if use_P:   # 使用泊松噪声作为shot noisy
        noisy_shot = np.random.poisson(MFM*y/p['K']).astype(np.float32) * p['K'] / MFM
    else:   # 不考虑shot noisy
        noisy_shot = y + np.random.randn(*y.shape).astype(np.float32) * np.sqrt(np.maximum(y/p['K'], 1e-10)) * p['K'] / MFM
    if not use_black:
        if use_TL:   # 使用TL噪声作为read noisy
            noisy_read = stats.tukeylambda.rvs(p['lam'], scale=p['sigTL']/MFM, size=y.shape).astype(np.float32)
        else:   # 使用高斯噪声作为read noisy
            noisy_read = stats.norm.rvs(scale=p['sigGs']/MFM, size=y.shape).astype(np.float32)
        # 行噪声需要使用行的维度h，[1,c,h,w]所以-2是h
        noisy_row = np.random.randn(y.shape[-3], y.shape[-2], 1).astype(np.float32) * p['sigR']/MFM if use_R else 0
        noisy_q = np.random.uniform(low=-0.5, high=0.5, size=y.shape) if use_Q else 0
        noisy_bias = p['bias'].reshape(-1,1,1) if use_D else 0
    else:
        noisy_read = 0
        noisy_row = 0
        noisy_q = 0
        noisy_bias = 0

    # 归一化回[0, 1]
    z = (noisy_shot + noisy_read + noisy_row + noisy_q + noisy_bias) / (p['wp'] - p['bl'])
    # 模拟实际情况
    z = np.clip(z, -p['bl']/p['wp'], 1) if not clip else np.clip(z, 0, 1)
    if ori is False:
        z = z * p['ratio']

    return z.astype(np.float32)

# @fn_timer
def generate_noisy_torch(y, camera_type=None,  noise_code='p', param=None, MultiFrameMean=1, ori=False, clip=False):
    p = param
    y = y * (p['wp'] - p['bl'])
    # p['ratio'] = 1/p['ratio'] # 临时行为，为了快速实现MFM
    y = y / p['ratio']
    MFM = MultiFrameMean ** 0.5

    use_R = True if 'r' in noise_code.lower() else False
    use_Q = True if 'q' in noise_code.lower() else False
    use_TL = True if 'g' in noise_code.lower() else False
    use_P = True if 'p' in noise_code.lower() else False
    use_D = True if 'd' in noise_code.lower() else False
    use_black = True if 'b' in noise_code.lower() else False

    if use_P:   # 使用泊松噪声作为shot noisy
        noisy_shot = tdist.Poisson(MFM*y/p['K']).sample() * p['K'] / MFM
    else:   # 不考虑shot noisy
        noisy_shot = tdist.Normal(y).sample() * torch.sqrt(torch.max(y/p['K'], 1e-10)) * p['K'] / MFM
    if not use_black:
        if use_TL:   # 使用TL噪声作为read noisy
            raise NotImplementedError
            # noisy_read = stats.tukeylambda.rvs(p['lam'], scale=p['sigTL'], size=y.shape).astype(np.float32)
        else:   # 使用高斯噪声作为read noisy
            noisy_read = tdist.Normal(loc=torch.zeros_like(y), scale=p['sigGs']/MFM).sample()
    else:
        noisy_read = 0
    # 使用行噪声
    noisy_row = torch.randn(y.shape[-3], y.shape[-2], 1, device=y.device) * p['sigR'] / MFM if use_R else 0
    noisy_q = (torch.rand(y.shape, device=y.device) - 0.5) * p['q'] * (p['wp'] - p['bl']) if use_Q else 0
    noisy_bias = torch.from_numpy(p['bias'].reshape(-1,1,1)) if use_D else 0

    # 归一化回[0, 1]
    z = (noisy_shot + noisy_read + noisy_row + noisy_q + noisy_bias) / (p['wp'] - p['bl'])
    # 模拟实际情况
    z = torch.clamp(z, -p['bl']/p['wp'], 1) if not clip else torch.clamp(z, 0, 1)
    # ori_brightness
    if ori is False:
        z = z * p['ratio']

    return z

class HighBitRecovery:
    def __init__(self, camera_type='IMX686', noise_code='prq', param=None, 
                perturb=True, factor=6, float=True):
        self.camera_type = camera_type
        self.noise_code = noise_code
        self.param = param
        self.perturb = perturb
        self.factor = factor
        self.float = float
        self.lut = {}
    
    def get_lut(self, iso_list, blc_mean=None):
        for iso in iso_list:
            if blc_mean is None:
                bias = 0
            else:
                bias = np.mean(blc_mean[iso])
            if self.perturb:
                sigma_t = 0.1
                bias += np.random.randn() * sigma_t
            self.lut[iso] = self.HB2LB_LUT(iso, bias)
    
    def HB2LB_LUT(self, iso, bias=0, param=None):
        # 标记LUT区间
        lut_info = {}
        # 获得标定的噪声参数
        p = sample_params_max(self.camera_type, iso=iso) if param is None else param
        lut_info['param'] = p
        # 选择一种分布，依照该分布映射到HighBit
        if 'g' in self.noise_code.lower():
            dist = stats.tukeylambda(p['lam'], loc=bias, scale=p['sigTL'])
            sigma = p['sigTL']
            lut_info['dist'] = dist
        else:
            dist = stats.norm(loc=bias, scale=p['sigGs'])
            sigma = p['sigGs']
            lut_info['dist'] = dist
        
        # 寻址范围为[-6σ,6σ]，离群点不做映射恢复
        low = max(int(-sigma*self.factor + bias), -p['bl']+1)
        high = int(sigma*self.factor + bias)
        for x in range(low, high):
            lut_info[x] = {
                # 累积概率函数的起点
                'cdf': dist.cdf(x-0.5),
                # 累积概率函数的变化范围
                'range': dist.cdf(x+0.5) - dist.cdf(x-0.5),
            }
        lut_info['low'] = low
        lut_info['high'] = high
        lut_info['bias'] = bias
        lut_info['sigma'] = sigma
        return lut_info
    
    def map(self, data, iso=6400, norm=True):    # 将LB图像映射成HB图像
        p = self.lut[iso]['param']
        if np.max(data) <= 1: data = data * (p['wp'] - p['bl'])
        data_float = data.copy()
        data = np.round(data_float)
        if self.float:
            delta = data_float - data
        rand = np.random.uniform(0, 1, size=data.shape)
        # 快速版：寻址范围为[-6σ,6σ]，离群点不做映射恢复
        for x in range(self.lut[iso]['low'], self.lut[iso]['high']):
            keys = (data==x)
            cdf = self.lut[iso][x]['cdf']
            r = self.lut[iso][x]['range']
            # 根据ppf反推分位点
            data[keys] = self.lut[iso]['dist'].ppf(cdf + rand[keys] * r)
        if self.float:
            data = data + delta
        if norm:
            data = data / (p['wp'] - p['bl'])
        else:
            data = data + p['bl']
        
        return data

class ELDEvalDataset(torch.utils.data.Dataset):
    # 初始化函数，传入参数：basedir（基础路径），camera_suffix（相机后缀），scenes（场景），img_ids（图像id）
    def __init__(self, basedir, camera_suffix=('NikonD850','.nef'), scenes=None, img_ids=None):
        super(ELDEvalDataset, self).__init__()
        self.basedir = basedir
        self.camera_suffix = camera_suffix # ('Canon', '.CR2')
        self.scenes = scenes
        self.img_ids = img_ids
        # self.input_dict = {}
        # self.target_dict = {}
        
    # 获取每一个图像的数据
    def __getitem__(self, i):
        camera, suffix = self.camera_suffix
        
        scene_id = i // len(self.img_ids)
        img_id = i % len(self.img_ids)

        scene ='scene-{}'.format(self.scenes[scene_id])

        datadir = os.path.join(self.basedir, camera, scene)

        input_path = os.path.join(datadir, 'IMG_{:04d}{}'.format(self.img_ids[img_id], suffix))

        gt_ids = np.array([1, 6, 11, 16])
        ind = np.argmin(np.abs(self.img_ids[img_id] - gt_ids))
        
        target_path = os.path.join(datadir, 'IMG_{:04d}{}'.format(gt_ids[ind], suffix))

        iso, expo = metainfo(target_path)
        target_expo = iso * expo
        iso, expo = metainfo(input_path)

        ratio = target_expo / (iso * expo)
        
        with rawpy.imread(input_path) as raw:
            input = pack_raw_bayer(raw) * ratio            

        with rawpy.imread(target_path) as raw:
            target = pack_raw_bayer(raw)

        input = np.maximum(np.minimum(input, 1.0), 0)
        target = np.maximum(np.minimum(target, 1.0), 0)
        input = np.ascontiguousarray(input)
        target = np.ascontiguousarray(target)        

        data = {'input': input, 'target': target, 'fn':input_path, 'rawpath': target_path}
        
        return data

    # 获取数据的长度
    def __len__(self):
        return len(self.scenes) * len(self.img_ids)

if __name__ == '__main__':
    path = 'F:/datasets/ELD/SonyA7S2/scene-8'
    files = [os.path.join(path, name) for name in os.listdir(path) if '.ARW' in name]
    for name in files:
        print(name)
        raw = rawpy.imread(name)
        img = raw.raw_image_visible.astype(np.float32)[np.newaxis,:,:]
        # img = img[:, 1000:1500, 2200:2700]
        fig = plt.figure(figsize=(16,10))
        img = np.clip((img-512) / (16383-512), 0, 1)
        p = sample_params_max(camera_type='SonyA7S2', ratio=100, iso=1600)

        noisy = generate_noisy_obs(img,camera_type='SonyA7S2', param=p)
        # refer_path = path+'\\'+'DSC02750.ARW'
        # raw_refer = rawpy.imread(refer_path)
        # print(np.min(raw_refer.raw_image_visible), np.max(raw_refer.raw_image_visible), np.mean(raw_refer.raw_image_visible))
        # raw_refer.raw_image_visible[:,:] = np.clip((raw_refer.raw_image_visible.astype(np.float32)-512) / (16383-512)*200, 0, 1)*16383
        # print(np.min(raw_refer.raw_image_visible), np.max(raw_refer.raw_image_visible), np.mean(raw_refer.raw_image_visible))
        # out1 = raw_refer.postprocess(use_camera_wb=True, no_auto_bright=True)
        # print(np.min(out1), np.max(out1), np.mean(out1))
        # plt.imsave('real.png', out1)
        # plt.imshow(out1)
        # plt.show()
        raw.raw_image_visible[:,:] = noisy[0,:,:]*16383
        out = raw.postprocess(use_camera_wb=True)
        plt.imshow(out)
        plt.show()
        plt.imsave('gen.png', out)
        print('test')
        print("")