import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch import nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CategoricalEncoder(nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, dropout, maxpool_indices, batch_norm=True, flatten=False):
        super(CategoricalEncoder, self).__init__()
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

class CategoricalDecoder(nn.Module):
    def __init__(self, num_var, num_latent, num_neurons, dropout, upsample_indices, batch_norm=True):
        super(CategoricalDecoder, self).__init__()
        self.num_var = num_var
        self.num_latent = num_latent
        self.num_neurons = num_neurons
        self.upsample_indices = [(3 + batch_norm) * (len(num_neurons) - x) - 2 for x in upsample_indices]
        self.num_neurons.reverse()
        self.dims = [28, 14, 7, 3, 1]

        num_units = [num_latent] + self.num_neurons

        layers = [nn.Linear(num_latent, (self.dims[len(self.upsample_indices)] ** 2) * num_latent)]

        for n_prev, n_new in zip(num_units[0:-1], num_units[1:]):
            layers.append(nn.ConvTranspose2d(n_prev, n_new, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            if batch_norm:
                layers.append(nn.BatchNorm2d(n_new))
        self.layers = nn.ModuleList(layers)
        self.probs = nn.Conv2d(num_neurons[-1], num_var, kernel_size=3, padding=1)
        self.probs_act = nn.Softmax(dim=1)

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

        h = self.probs(h)
        probs = self.probs_act(h).permute(0,3,2,1)

        m = Categorical(probs)
        z = m.sample()

        # Translate bin-assignments to numerical values using midpoint
        z = (z/5)+0.1

        return z, probs


class CategoricalVAE(nn.Module):
    def __init__(self, encoder, decoder):
        super(CategoricalVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        z, mu_enc, log_sigma_enc = self.encoder(x)
        x_reconstr, probs = self.decoder(z)
        return x_reconstr

    def sample(self, n):
        noise = torch.rand((n, self.encoder.num_latent)).to(device)
        return self.decoder(noise)[0]


class CategoricalELBOLoss(nn.Module):
    def __init__(self):
        super(CategoricalELBOLoss, self).__init__()
        self.epsilon = 1e-16

    def forward(self, x, probs, enc_mu, enc_var):
        probs = probs.reshape(-1, 5)
        x = x.reshape(-1)
        likelihoods = probs[range(len(x) ), x]
        log_p = torch.sum(torch.log(likelihoods+self.epsilon))
        KL = -0.5 * torch.sum(1 + enc_var - (enc_mu ** 2) - torch.exp(enc_var))
        return -log_p + KL, log_p, KL