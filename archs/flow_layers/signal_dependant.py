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
from archs.flow_layers.neural_spline import (unconstrained_rational_quadratic_spline, \
                                            rational_quadratic_spline, sum_except_batch)

class SignalDependantISO(nn.Module):
    def __init__(self, device='cpu', name='g_iso'):
        super().__init__()
        self.name = name
        self.legal_iso = torch.tensor([50, 64, 80, 100, 125, 160, 200, 250, 320, 400, 500, 640, 800, 1000, 1250, 1600] +\
            [2000, 2500, 3200, 4000, 5000, 6400, 8000, 10000, 12800, 16000, 20000, 25600, 32000, 40000, 51200], dtype=torch.float32, device=device)
        self.cam_param = nn.Parameter(torch.tensor(np.zeros((len(self.legal_iso), 3)), dtype=torch.float32, device=device), requires_grad=False)
        self.gain = nn.Parameter(torch.tensor(-6.0, dtype=torch.float32, device=device), requires_grad=True)
        self.beta1 = nn.Parameter(torch.tensor(-5.0, dtype=torch.float32, device=device), requires_grad=True)
        self.beta2 = nn.Parameter(torch.tensor(-4.0, dtype=torch.float32, device=device), requires_grad=True)

    def to_device(self, device='cuda'):
        self.legal_iso = self.legal_iso.to(device)
        self.cam_param = self.cam_param.to(device)
        self.gain = self.gain.to(device)
        self.beta1 = self.beta1.to(device)
        self.beta2 = self.beta2.to(device)
    
    def _scale(self, clean, iso):
        # ISO 索引
        l = torch.searchsorted(self.legal_iso, iso, right=False)
        r = torch.searchsorted(self.legal_iso, iso, right=True)
        iso_l, iso_r = self.legal_iso[l], self.legal_iso[r]
        cam_param_l, cam_param_r = torch.exp(self.cam_param[l]), torch.exp(self.cam_param[r])
        cam_param = ((iso - iso_l) * cam_param_r + (iso_r - iso) * cam_param_l) / (iso_r - iso_l) if iso_r - iso_l != 0 else cam_param_l
        # Beta1, Beta2, Gain 计算
        beta1 = torch.exp(self.beta1 * cam_param[0])
        beta2 = torch.exp(self.beta2 * cam_param[1])
        gain = torch.exp(self.gain * cam_param[2]) * iso
        # 噪声缩放: 根据计算得到的参数，对输入的 clean 图像进行噪声缩放操作，并确保缩放后的值非负。
        scale = beta1 * clean / gain + beta2
        assert torch.min(scale) >= 0
        return torch.sqrt(scale)

    def _inverse(self, z, **kwargs):
        self.to_device(z.device)
        scale = self._scale(kwargs['clean'], kwargs['iso'])
        x = z * scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None
    
        self.to_device(x.device)
        scale = self._scale(kwargs['clean'], kwargs['iso'])

        if writer:
            writer.add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), step)

        z = x / scale

        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])

        return z, log_abs_det_J_inv

class SignalDependant(nn.Module):
    def __init__(self, scale, param_inits=False, name='sdn'):
        super(SignalDependant, self).__init__()
        self.name = name
        self.param_inits = param_inits
        self._scale = scale(self.param_inits)

    def _inverse(self, z, **kwargs):
        scale = self._scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])
        x = z * scale
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        scale = self._scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_mean', torch.mean(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_min', torch.min(scale), kwargs['step'])
            kwargs['writer'].add_scalar('model/' + self.name + '_scale_max', torch.max(scale), kwargs['step'])

        z = x / scale
        log_abs_det_J_inv = - torch.sum(torch.log(scale), dim=[1, 2, 3])
        return z, log_abs_det_J_inv

class SignalDependantExp2(nn.Module):
    def __init__(self, log_scale, gain_scale, param_inits=False, device='cpu', name='sdn'):
        super(SignalDependantExp2, self).__init__()
        self.name = name
        self.param_inits = param_inits
        self._log_scale = log_scale(gain_scale, self.param_inits, device=device, name='sdn_layer_gain_scale')

    def _inverse(self, z, **kwargs):
        log_scale = self._log_scale(kwargs['clean'], kwargs['iso'], kwargs['cam'])
        x = z * torch.exp(log_scale)
        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        writer = kwargs['writer'] if 'writer' in kwargs.keys() else None
        step = kwargs['step'] if 'step' in kwargs.keys() else None
        
        log_scale = self._log_scale(kwargs['clean'], kwargs['iso'], kwargs['cam'], writer, step)

        if 'writer' in kwargs.keys():
            writer.add_scalar('model/' + self.name + '_log_scale_mean', torch.mean(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_min', torch.min(log_scale), step)
            writer.add_scalar('model/' + self.name + '_log_scale_max', torch.max(log_scale), step)

        z = x / torch.exp(log_scale)
        log_abs_det_J_inv = - torch.sum(log_scale, dim=[1, 2, 3])
        return z, log_abs_det_J_inv


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3

class SignalDependantNS(nn.Module):
    def __init__(
        self,
        transform_net,
        x_shape,
        param_inits=False,
        num_bins=10,
        tails="linear",
        tail_bound=1.0,
        name='sdn',
        device='cpu',
        min_bin_width=DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=DEFAULT_MIN_DERIVATIVE,
        ):
        super(SignalDependantNS, self).__init__()
        self.name = name
        self.ic, self.i0, self.i1 = x_shape
        self.num_bins = num_bins
        self.tails = tails
        self.tail_bound = tail_bound

        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.min_derivative = min_derivative
        
        # self._transform_net = transform_net(
        #     x_shape=x_shape,
        #     width=16,
        #     num_in=x_shape[0],
        #     num_output=x_shape[0] * self._transform_dim_multiplier(),
        #     device=device
        # )

        self._transform_net = transform_net(
            x_shape[0],
            x_shape[0] * self._transform_dim_multiplier()
        )

    def _transform_dim_multiplier(self):
        if self.tails == "linear":
            return self.num_bins * 3 - 1
        else:
            return self.num_bins * 3 + 1

    def _inverse(self, z, **kwargs):
        b, c, h, w = z.shape
        transform_params = self._transform_net(kwargs['clean'])
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self._transform_net, 'width'):
            unnormalized_widths /= np.sqrt(self._transform_net.width)
            unnormalized_heights /= np.sqrt(self._transform_net.width)
        elif hasattr(self._transform_net, 'hidden_channels'):
            unnormalized_widths /= np.sqrt(self._transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self._transform_net.hidden_channels)
        else:
            warnings.warn('Inputs to the softmax are not scaled down: initialization might be bad.')

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        x, logabsdet = spline_fn(
            inputs=z,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=True,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        logabsdet = sum_except_batch(logabsdet)

        return x

    def _forward_and_log_det_jacobian(self, x, **kwargs):
        b, c, h, w = x.shape
        transform_params = self._transform_net(kwargs['clean'])
        transform_params = transform_params.reshape(b, c, -1, h, w).permute(
                0, 1, 3, 4, 2
            )
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins : 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins :]

        if hasattr(self._transform_net, 'width'):
            unnormalized_widths /= np.sqrt(self._transform_net.width)
            unnormalized_heights /= np.sqrt(self._transform_net.width)
        elif hasattr(self._transform_net, 'hidden_channels'):
            unnormalized_widths /= np.sqrt(self._transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self._transform_net.hidden_channels)
        else:
            warnings.warn('Inputs to the softmax are not scaled down: initialization might be bad.')

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        z, logabsdet = spline_fn(
            inputs=x,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=False,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            **spline_kwargs
        )

        logabsdet = sum_except_batch(logabsdet)

        return z, logabsdet