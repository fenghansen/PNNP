from .utils import *

def scale_down(img):
    return np.float32(img) / 255.

def scale_up(img):
    return np.uint8(img.clip(0,1) * 255.)

def tensor2im(image_tensor, visualize=False, video=False):    
    image_tensor = image_tensor.detach()

    if visualize:                
        image_tensor = image_tensor[:, 0:3, ...]

    if not video: 
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:
        image_numpy = image_tensor.cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (0, 2, 3, 1))) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy

def quality_assess(X, Y, data_range=255):
    # Y: correct; X: estimate
    if X.ndim == 3:
        psnr = compare_psnr(Y, X, data_range=data_range)
        ssim = compare_ssim(Y, X, data_range=data_range, channel_axis=-1)
        return {'PSNR':psnr, 'SSIM': ssim}
    else:
        raise NotImplementedError

def feature_vis(tensor, name='out', save=False):
    feature = tensor.detach().cpu().numpy().transpose(0,2,3,1)
    if save:
        if feature.min() < 0 or feature.max()>1:
            warnings.warn('Signals are clipped to [0, 1] for visualization!!!!')
        os.makedirs('worklog/feature_vis', exist_ok=True)
        if feature.shape[-1] == 4:
            feature = feature[..., :3]
        for i in range(len(feature)):
            cv2.imwrite(f'worklog/feature_vis/{name}_{i}.png', np.uint8(feature[i,:,:,::-1]*255))
    return feature

def plot_sample(img_lr, img_dn, img_hr, filename='result', model_name='Unet', 
                epoch=-1, print_metrics=False, save_plot=True, save_path='./', res=None):
    if np.max(img_hr) <= 1:
        # 变回uint8
        img_lr = scale_up(img_lr)
        img_dn = scale_up(img_dn)
        img_hr = scale_up(img_hr)
    # 计算PSNR和SSIM
    if res is None:
        psnr = []
        ssim = []
        psnr.append(compare_psnr(img_hr, img_lr))
        psnr.append(compare_psnr(img_hr, img_dn))
        ssim.append(compare_ssim(img_hr, img_lr, channel_axis=-1))
        ssim.append(compare_ssim(img_hr, img_dn, channel_axis=-1))
        psnr.append(-1)
        ssim.append(-1)
    else:
        psnr = [res[0], res[2], -1]
        ssim = [res[1], res[3], -1]
    # Images and titles
    images = {
        'Noisy Image': img_lr,
        model_name: img_dn,
        'Ground Truth': img_hr
    }
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    # Plot the images. Note: rescaling and using squeeze since we are getting batches of size 1
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    for i, (title, img) in enumerate(images.items()):
        axes[i].imshow(img)
        axes[i].set_title("{} - {} - psnr:{:.2f} - ssim{:.4f}".format(title, img.shape, psnr[i], ssim[i]))
        axes[i].axis('off')
    plt.suptitle('{} - Epoch: {}'.format(filename, epoch))
    if print_metrics:
        log('PSNR:', psnr)
        log('SSIM:', ssim)
    # Save directory
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    savefile = os.path.join(save_path, "{}-Epoch{}.jpg".format(filename, epoch))
    if save_plot:
        denoisedfile = os.path.join(save_path, "{}_denoised.png".format(filename))
        cv2.imwrite(denoisedfile, img_dn[:,:,::-1])
        fig.savefig(savefile, bbox_inches='tight')
        plt.close()
    return psnr, ssim, filename

def save_picture(img_sr, save_path='./images/test',frame_id='0000'):
    # 变回uint8
    img_sr = scale_up(img_sr.transpose(1,2,0))
    if os._exists(save_path) is not True:
        os.makedirs(save_path, exist_ok=True)
    plt.imsave(os.path.join(save_path, frame_id+'.png'), img_sr)
    gc.collect()