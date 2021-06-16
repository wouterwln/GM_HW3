import torch
import torch.nn.functional as F
from torch.distributions import Beta
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BetaEncoder(nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, dropout, maxpool_indices, batch_norm=True, flatten=False):
        super(BetaEncoder, self).__init__()
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
        h = x
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

class BetaDecoder(nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, dropout, upsample_indices, batch_norm=True):
        super(BetaDecoder, self).__init__()
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
        self.log_alpha = nn.Conv2d(num_neurons[-1], num_var, 3, padding=1)
        self.alpha_act = nn.Tanh()
        self.log_beta = nn.Conv2d(num_neurons[-1], num_var, 3, padding=1)
        self.beta_act = nn.Tanh()

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

        log_alpha = self.log_alpha(h)
        log_alpha = self.alpha_act(self.log_alpha(h))
        log_beta = self.log_beta(h)
        log_beta = self.beta_act(self.log_beta(h))

        m = Beta(torch.exp(log_alpha), torch.exp(log_beta))
        z = m.sample()

        return z, log_alpha, log_beta


class BetaVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(BetaVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mu_enc, log_sigma_enc = self.encoder(x)
        x_reconstr, log_alpha_dec, log_beta_dec = self.decoder(z)
        return x_reconstr

    def sample(self, n):
        noise = torch.rand((n, self.encoder.num_latent)).to(device)
        return self.decoder(noise)[0]


class BetaELBOLoss(nn.Module):
    def __init__(self):
        super(BetaELBOLoss, self).__init__()
        self.epsilon = 1e-16

    def forward(self, x, dec_alpha, dec_beta, enc_mu, enc_var):
        dec_alpha = torch.exp(dec_alpha)
        dec_beta = torch.exp(dec_beta)
        B_ab = torch.lgamma(dec_alpha) + torch.lgamma(dec_beta) - torch.lgamma(dec_alpha + dec_beta)
        log_p = torch.sum((dec_alpha - 1) * torch.log(x+self.epsilon) + (dec_beta - 1) * torch.log((1-x)+self.epsilon) - B_ab)
        KL = -0.5 * torch.sum(1 + enc_var - (enc_mu ** 2) - torch.exp(enc_var))
        return -log_p + KL, log_p, KL