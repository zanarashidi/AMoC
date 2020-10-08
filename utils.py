import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.optimizer import Optimizer, required

class AE_MNIST(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer1 = nn.Linear(
            in_features=kwargs["input_shape"], out_features=1000)
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=1000, out_features=500)
        self.encoder_hidden_layer3 = nn.Linear(
            in_features=500, out_features=250)
        self.encoder_output_layer = nn.Linear(
            in_features=250, out_features=30)
        self.decoder_hidden_layer1 = nn.Linear(
            in_features=30, out_features=250)
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=250, out_features=500)
        self.decoder_hidden_layer3 = nn.Linear(
            in_features=500, out_features=1000)
        self.decoder_output_layer = nn.Linear(
            in_features=1000, out_features=kwargs["input_shape"])

    def forward(self, features):
        activation1 = self.encoder_hidden_layer1(features)
        activation1 = torch.relu(activation1)
        activation2 = self.encoder_hidden_layer2(activation1)
        activation2 = torch.relu(activation2)
        activation3 = self.encoder_hidden_layer3(activation2)
        activation3 = torch.relu(activation3)
        code = self.encoder_output_layer(activation3)
        code = torch.relu(code)
        activation4 = self.decoder_hidden_layer1(code)
        activation4 = torch.relu(activation4)
        activation5 = self.decoder_hidden_layer2(activation4)
        activation5 = torch.relu(activation5)
        activation6 = self.decoder_hidden_layer3(activation5)
        activation6 = torch.relu(activation6)
        activation7 = self.decoder_output_layer(activation6)
        reconstructed = torch.relu(activation7)
        return reconstructed

class AMoC(Optimizer):
    dotp = 0.0
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, beta=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, beta=beta)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(AMoC, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AMoC, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            beta = group['beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    self.dotp = torch.sum(torch.mul(buf/torch.norm(buf), d_p/torch.norm(d_p))) #dot product
                    buf.mul_(momentum*(1-beta*self.dotp)).add_(1 - dampening, d_p) #adaptive coefficient

                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss
