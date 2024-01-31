"""
Copyright (c) 2022 Samsung Electronics Co., Ltd.

Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) License, (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc/4.0
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
For conditions of distribution and use, see the accompanying LICENSE.md file.

"""
import torch
from torch import nn
import numpy as np

class Gain(nn.Module):
    def __init__(self, name='gain', device='cuda'):
        super(Gain, self).__init__()
        self.scale = nn.Parameter(torch.tensor(1.0, device=device), requires_grad=True)
        self.name = name

    def _inverse(self, z, **kwargs):
        x = z * self.scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        scale = self.scale + (x * 0.0)

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_min', torch.min(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_max', torch.max(scale), kwargs['step'])

        z = x / scale
        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class GainExp2(nn.Module):
    def __init__(self, gain_scale, param_inits, device='cpu', name='gain'):
        super(GainExp2, self).__init__()
        self.name = name
        self._gain_scale = gain_scale(param_inits, device=device, name='gain_layer_gain_scale')

    def _inverse(self, z, **kwargs):
        scale, _ = self._gain_scale(kwargs['iso'])
        x = z * scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None
    
        scale, _ = self._gain_scale(kwargs['iso'], writer, step)

        if writer:
            writer.add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), step)

        z = x / scale

        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])

        return z, log_abs_det_J_inv

class GainISO(nn.Module):
    def __init__(self, device='cpu', name='g_iso'):
        super(GainISO, self).__init__()
        self.name = name
        self.legal_iso = torch.tensor([50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600] +\
            [2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000, 51200], dtype=torch.float32, device=device)
        self.cam_param = nn.Parameter(torch.tensor(np.zeros_like(self.legal_iso), dtype=torch.float32, device=device), requires_grad=True)
        self.gain_params = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32, device=device), requires_grad=True)

    def to_device(self, device='cuda'):
        self.legal_iso = self.legal_iso.to(device)
        self.cam_param = self.cam_param.to(device)
        self.gain_params = self.gain_params.to(device)
    
    def _scale(self, iso):
        l = torch.searchsorted(self.legal_iso, iso, right=False)
        r = torch.searchsorted(self.legal_iso, iso, right=True)
        iso_l, iso_r = self.legal_iso[l], self.legal_iso[r]
        cam_param_l, cam_param_r = torch.exp(self.cam_param[l]), torch.exp(self.cam_param[r])
        cam_param = ((iso - iso_l) * cam_param_r + (iso_r - iso) * cam_param_l) / (iso_r - iso_l) if iso_r - iso_l != 0 else cam_param_l
        scale = torch.exp(cam_param * self.gain_params) * iso
        return scale

    def _inverse(self, z, **kwargs):
        iso = kwargs['iso']
        self.to_device(z.device)
        scale = self._scale(iso)
        x = z * scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None
    
        iso = kwargs['iso']
        self.to_device(x.device)
        scale = self._scale(iso) + (x * 0.0)

        if writer:
            writer.add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), step)

        z = x / scale

        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])

        return z, log_abs_det_J_inv
