import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderTrained(nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, dropout, maxpool_indices, batch_norm=True, flatten=False):
        super(EncoderTrained, self).__init__()
        self.num_var = num_var
        self.num_latent = num_latent
        self.num_neurons = num_neurons
        self.maxpool_indices = [(3 + batch_norm) * x for x in maxpool_indices]
        self.dims = [28, 14, 7, 3, 1]

        num_units = [num_var] + num_neurons
        layers = []
        for n_prev, n_new in zip(num_units[0:-1], num_units[1:]):
            layers.append(nn.Conv2d(n_prev, n_new, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if batch_norm:
                layers.append(nn.BatchNorm2d(n_new))
        self.flatten = flatten

        self.mu = nn.Linear((self.dims[len(maxpool_indices)] ** 2) * num_neurons[-1], num_latent)
        self.log_var = nn.Linear((self.dims[len(maxpool_indices)] ** 2) * num_neurons[-1], num_latent)
        self.layers = nn.ModuleList(layers)
        self.var_act = nn.Softplus()

    def forward(self, x):
        h = x.to(device)
        for i, layer in enumerate(self.layers):
            if i in self.maxpool_indices:
                h = F.max_pool2d(layer(h), 2)
            else:
                h = layer(h)
        h = h.view(h.shape[0], -1)
        mu = self.mu(h)
        log_sigma = self.var_act(self.log_var(h))
        z = self.reparameterization(mu, log_sigma)

        return z, mu, log_sigma

    def reparameterization(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        epsilon = torch.rand_like(sigma).to(device)
        z = mu + sigma * epsilon

        return z

class Encoder(nn.Module):
    def __init__(self, num_latent):
        super(Encoder, self).__init__()
        self.mu = nn.Parameter(torch.zeros(num_latent))
        self.sigma = nn.Parameter(torch.ones(num_latent))

    def forward(self):
        z = self.reparameterization(self.mu, self.sigma)

        return z, self.mu, self.sigma

    def reparameterization(self, mu, sigma):
        epsilon = torch.rand_like(sigma).to(device)
        z = mu.to(device) + sigma.to(device) * epsilon

        return z


class Decoder(nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, dropout, upsample_indices, batch_norm=True, constant_var=True):
        super(Decoder, self).__init__()
        self.num_var = num_var
        self.num_latent = num_latent
        self.num_neurons = num_neurons
        self.upsample_indices = [(3 + batch_norm) * (len(num_neurons) - x) - 2 for x in upsample_indices]
        self.num_neurons.reverse()
        self.dims = [28, 14, 7, 3, 1]

        num_units = [num_latent] + self.num_neurons

        layers = [nn.Linear(num_latent, (self.dims[len(self.upsample_indices)] ** 2) * num_latent)]

        for n_prev, n_new in zip(num_units[0:-1], num_units[1:]):
            layers.append(nn.ConvTranspose2d(n_prev, n_new, 3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if batch_norm:
                layers.append(nn.BatchNorm2d(n_new))
        self.layers = nn.ModuleList(layers)
        self.mu = nn.ConvTranspose2d(num_neurons[-1], num_var, 3, padding=1)
        if not constant_var:
            self.log_var = nn.ConvTranspose2d(num_neurons[-1], num_var, 3, padding=1)
            self.var_act = nn.Softplus(beta=0.5)
        self.constant_var = constant_var

    def forward(self, x):
        h = x
        h = self.layers[0](h)
        h = h.view(h.shape[0], self.num_latent, self.dims[len(self.upsample_indices)],
                   self.dims[len(self.upsample_indices)])
        for i, layer in enumerate(self.layers[1:]):
            if i in self.upsample_indices:
                index = self.upsample_indices.index(i)
                h = F.interpolate(layer(h), self.dims[index])
            else:
                h = layer(h)

        mu = self.mu(h)
        if not self.constant_var:
            log_sigma = self.var_act(self.log_var(h))
        else:
            log_sigma = torch.log(torch.tensor(0.05))
        z = self.reparameterization(mu, log_sigma)

        return z, mu, log_sigma

    def reparameterization(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        epsilon = torch.rand_like(sigma).to(device)
        z = mu + sigma * epsilon

        return z


class VAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mu_enc, log_sigma_enc = self.encoder(x)
        x_reconstr, mu_dec, log_sigma_dec = self.decoder(z)
        return x_reconstr

    def sample(self, n):
        noise = torch.rand((n, self.encoder.num_latent)).to(device)
        return self.decoder(noise)[1]


class ELBOLoss(nn.Module):
    def __init__(self):
        super(ELBOLoss, self).__init__()
        self.pi = (2 * torch.acos(torch.zeros(1))).to(device)

    def forward(self, x, dec_mu, dec_var, enc_mu, enc_var):
        dec_mu = dec_mu[:, :, :, :14]
        enc_var = torch.log(enc_var)
        reconstr_loss = F.mse_loss(x, dec_mu, reduction="none")
        log_p = -0.5 * torch.sum(torch.log(2 * self.pi) + dec_var + (reconstr_loss / (torch.exp(dec_var) + 1e-16)))
        KL = -0.5 * torch.sum(1 + enc_var - (enc_mu ** 2) - torch.exp(enc_var))
        return -log_p + KL, log_p, KL