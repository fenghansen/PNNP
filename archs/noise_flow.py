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

from archs.flow_layers.conv2d1x1 import Conv2d1x1
from archs.flow_layers.affine_coupling import AffineCoupling, ShiftAndLogScale
from archs.flow_layers.signal_dependant import SignalDependant, SignalDependantISO
from archs.flow_layers.gain import Gain, GainISO
from archs.flow_layers.utils import SdnModelScale
# from archs.flow_layers.linear_transformation import LinearTransformation

class NoiseFlow(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.default_args(args)
        self.x_shape = self.args['x_shape']
        self.arch = self.args['arch']
        self.flow_permutation = self.args['flow_permutation']
        self.param_inits = self.args['param_inits']
        self.decomp = self.args['lu_decomp']
        self.model = nn.ModuleList(self.noise_flow_arch(self.x_shape))

    def default_args(self, args=None):
        self.args = {}
        self.args['x_shape'] = (4, 256, 256)
        self.args['arch'] = 'sdn|unc|unc|unc|unc|gain|unc|unc|unc|unc'
        self.args['flow_permutation'] = 1
        self.args['param_inits'] = None
        self.args['lu_decomp'] = True
        if args is not None:
            for key in args:
                self.args[key] = args[key]

    def noise_flow_arch(self, x_shape):
        arch_lyrs = self.arch.split('|')  # e.g., unc|sdn|unc|gain|unc
        bijectors = []
        for i, lyr in enumerate(arch_lyrs):
            is_last_layer = False

            if lyr == 'unc':
                if self.flow_permutation == 0:
                    pass
                elif self.flow_permutation == 1:
                    print('|-Conv2d1x1')
                    bijectors.append(
                        Conv2d1x1(
                            num_channels=x_shape[0],
                            LU_decomposed=self.decomp,
                            name='Conv2d_1x1_{}'.format(i)
                        )
                    )
                else:
                    print('|-No permutation specified. Not using any.')
                    # raise Exception("Flow permutation not understood")

                print('|-AffineCoupling')
                bijectors.append(
                    AffineCoupling(
                        x_shape=x_shape,
                        shift_and_log_scale=ShiftAndLogScale,
                        name='unc_%d' % i
                    )
                )
            # elif lyr == 'lt':
            #     print('|-LinearTransfomation')
            #     bijectors.append(
            #         LinearTransformation(
            #             name='lt_{}'.format(i),
            #             device='cuda'
            #         )
            #     )
            # 我基于SID修改了一下
            # elif lyr == 'sdn':
            #     print('|-SignalDependant')
            #     bijectors.append(
            #         SignalDependant(
            #             name='sdn_%d' % i,
            #             scale=SdnModelScale,
            #             param_inits=self.param_inits
            #         )
            #     )
            elif lyr == 'sdn':
                print('|-SignalDependantISO')
                bijectors.append(
                    SignalDependantISO(name='sdn_%d' % i)
                )
            # 此处Noise2NoiseFlow的实现明显有问题，不支持ISO可变
            # elif lyr == 'gain':
            #     print('|-Gain')
            #     bijectors.append(
            #         Gain(name='gain_%d' % i)
            #     )
            elif lyr == 'giso':
                print('|-GainISO')
                bijectors.append(
                    GainISO(name='giso_%d' % i)
                )

        return bijectors

    def forward(self, **kwargs):
        if 'mode' in kwargs.keys() and kwargs['mode']!='forward':
            if kwargs['mode'] == 'sample':
                return self.sample(**kwargs)
            elif kwargs['mode'] == 'loss':
                return self.loss(**kwargs)
            elif kwargs['mode'] == 'inverse':
                return self.inverse(**kwargs)
        else:
            x = kwargs['noise']
            z = x
            objective = torch.zeros(x.shape[0], dtype=torch.float32, device=x.device)
            for bijector in self.model:
                z, log_abs_det_J_inv = bijector._forward_and_log_det_jacobian(z, **kwargs)
                objective += log_abs_det_J_inv

                if 'writer' in kwargs.keys():
                    kwargs['writer'].add_scalar('model/' + bijector.name, torch.mean(log_abs_det_J_inv), kwargs['step'])
            return z, objective

    def _loss(self, **kwargs):
        kwargs['mode'] = 'forward'
        x = kwargs['noise']
        z, objective = self.forward(**kwargs)
        # base measure
        logp, _ = self.prior("prior", x)

        log_z = logp(z)
        objective += log_z

        if 'writer' in kwargs.keys():
            kwargs['writer'].add_scalar('model/log_z', torch.mean(log_z), kwargs['step'])
            kwargs['writer'].add_scalar('model/z', torch.mean(z), kwargs['step'])
        nobj = - objective
        # std. dev. of z
        mu_z = torch.mean(x, dim=[1, 2, 3])
        var_z = torch.var(x, dim=[1, 2, 3])
        sd_z = torch.mean(torch.sqrt(var_z))

        return nobj, sd_z

    def loss(self, **kwargs):
        x = kwargs['noise']
        batch_average = torch.mean(x, dim=0)
        if 'writer' in kwargs.keys():
            kwargs['writer'].add_histogram('real_noise', batch_average, kwargs['step'])
            kwargs['writer'].add_scalar('real_noise_std', torch.std(batch_average), kwargs['step'])

        nll, sd_z = self._loss(**kwargs)
        nll_dim = torch.mean(nll) / np.prod(x.shape[1:])
        # nll_dim = torch.mean(nll)      # The above line should be uncommented

        return nll_dim, sd_z

    def inverse(self, **kwargs):
        x = kwargs['noise']
        for bijector in reversed(self.model):
            x = bijector._inverse(x, **kwargs)
        return x
    
    def sample(self, **kwargs):
        if 'clean' not in kwargs.keys():
            kwargs['clean'] = kwargs['noise']
        _, sample = self.prior("prior", kwargs['clean'])
        if 'eps_std' not in kwargs.keys(): eps_std = None
        z = sample(eps_std)
        x = z
        # inverse
        for bijector in reversed(self.model):
            x = bijector._inverse(x, **kwargs)
        batch_average = torch.mean(x, dim=0)
        if 'writer' in kwargs.keys():
            # kwargs['writer'].add_histogram('sample_noise', batch_average, kwargs['step'])
            kwargs['writer'].add_scalar('sample_noise_std', torch.std(batch_average), kwargs['step'])

        return x

    def prior(self, name, x):
        n_z = x.shape[1]
        h = torch.zeros([x.shape[0]] +  [2 * n_z] + list(x.shape[2:4]), device=x.device)
        pz = gaussian_diag(h[:, :n_z, :, :], h[:, n_z:, :, :])

        def logp(z1):
            objective = pz.logp(z1)
            return objective

        def sample(eps_std=None):
            if eps_std is not None:
                z = pz.sample2(pz.eps * torch.reshape(eps_std, [-1, 1, 1, 1]))
            else:
                z = pz.sample
            return z

        return logp, sample

def gaussian_diag(mean, logsd):
    class o(object):
        pass

    o.mean = mean
    o.logsd = logsd
    o.eps = torch.normal(torch.zeros(mean.shape, device=mean.device), torch.ones(mean.shape, device=mean.device))
    o.sample = mean + torch.exp(logsd) * o.eps
    o.sample2 = lambda eps: mean + torch.exp(logsd) * eps

    o.logps = lambda x: -0.5 * (np.log(2 * np.pi) + 2. * o.logsd + (x - o.mean) ** 2 / torch.exp(2. * o.logsd))
    o.logp = lambda x: torch.sum(o.logps(x), dim=[1, 2, 3])
    o.get_eps = lambda x: (x - mean) / torch.exp(logsd)
    return o
