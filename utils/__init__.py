from .utils import *
from .isp_ops import *
from .isp_algos import *
from .video_ops import *
from .visualization import *
from .kld_div import *

class AlgoDebugger():
    def __init__(self, args={}):
        self.default_args()
        for key in args:
            self.args[key] = args[key]

    def default_args(self):
        self.args = {}
        self.args['algo'] = FastGuidedFilter
        self.args['win_name'] = 'Show'
        self.args['trackbar'] = {
            'd': {'default': 5, 'max_num':50, 'func':lambda x:x//2*2+1},
            'eps': {'default': 20, 'max_num':80, 'func':lambda x:10**-(x/10)}
        }
 
    def nothing(self):
        pass

    def debug(self, imgs, params=None):
        algo_func = self.args['algo']
        win_name = self.args['win_name']
        tb = self.args['trackbar']
        # 创建窗口
        cv2.namedWindow(win_name)
        # 绑定老窗口和滑动条（滑动条的数值）
        for var in tb:
            cv2.createTrackbar(var, win_name, tb[var]['default'], tb[var]['max_num'], self.nothing)
        flag = 1
        img_ori = imgs[0]
        while True:
            p = {}
            # 提取滑动条的数值
            for var in tb:
                trans_func = tb[var]['func']
                p[var] = trans_func(cv2.getTrackbarPos(var, win_name))
            denoised = algo_func(*imgs, *p)
            if flag:
                result = denoised[:,:,:3]
            else:
                result = img_ori[:,:,:3]
            cv2.imshow(win_name, result)
            # 设置推出键
            k = cv2.waitKey(1)
            if k == ord('f'):
                flag = 1 - flag
            elif k == ord('q'):
                break
        # 关闭窗口
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # pth_transfer('/data/ELD/checkpoints/sid-paired/model_200_00280000.pt', 'checkpoints/SonyA7S2_Paired_Official_last_model')
    # names = [name for name in os.listdir('checkpoints') if 'best_model' in name]
    # for name in tqdm(names):
        # pth_transfer(dst_path=f'checkpoints/{name}', reverse=True)
    root_dir = '/data/SonyA7S2/resources-2087'
    ds_files = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if name[0] == 'd' and name[-4:]=='.npy']
    # bpc_files = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if name[0] == 'b' and name[-4:]=='.npy']
    isos = []
    legal_iso = np.array([50, 64, 80, 8000, 10000, 12800, 16000, 20000, 25600])
    pbar = tqdm(ds_files)
    for file in pbar:
        iso = int(os.path.basename(file)[16:-4])
        isos.append(iso)
        # if iso in legal_iso:
        pbar.set_description_str(f'ISO-{iso}')
        ds_denoised = row_denoise(file,iso)
        np.save(file, ds_denoised)
    print(sorted(isos))