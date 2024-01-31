from .utils import *

def frame_index_splitor(nframes=1, pad=True, reflect=True):
    # [b, 7, c, h ,w]
    r = nframes // 2
    length = 7 if pad else 8-nframes
    frames = []
    for i in range(length):
        frames.append([None]*nframes)
    if pad:
        for i in range(7):
            for k in range(nframes):
                frames[i][k] = i+k-r
    else:
        for i in range(8-nframes):
            for k in range(nframes):
                frames[i][k] = i+k
    if reflect:
        frames = num_reflect(frames,0,6)
    else:
        frames = num_clip(frames, 0, 6)
    return frames

def multi_frame_loader(frames ,index, gt=False, keepdims=False):
    loader = []
    for ind in index:
        imgs = []
        if gt:
            r = len(index[0]) // 2
            tensor = frames[:,ind[r],:,:,:]
            if keepdims:
                tensor = tensor.unsqueeze(dim=1)
        else:
            for i in ind:
                imgs.append(frames[:,i,:,:,:])
                tensor = torch.stack(imgs, dim=1)
        loader.append(tensor)
    return torch.stack(loader, dim=0)

def num_clip(nums, mininum, maxinum):
    nums = np.array(nums)
    nums = np.clip(nums, mininum, maxinum)
    return nums

def num_reflect(nums, mininum, maxinum):
    nums = np.array(nums)
    nums = np.abs(nums-mininum)
    nums = maxinum-np.abs(maxinum-nums)
    return nums