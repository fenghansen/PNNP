from .base_loss import *
# from .flow_loss import *

def PSNR_Loss(low, high):
    # 默认已归一化到[0,1]！！！！
    shape = low.shape
    if len(shape) <= 3:
        psnr = -10.0 * torch.log(torch.mean(torch.pow(high-low, 2))) / torch.log(torch.as_tensor(10.0))
    else:
        psnr = torch.zeros(shape[0])
        for i in range(shape[0]):
            psnr[i]=-10.0 * torch.log(torch.mean(torch.pow(high[i]-low[i], 2))) / torch.log(torch.as_tensor(10.0))
        # print(psnr)
        psnr = torch.mean(psnr)# / shape[0]
    return psnr 