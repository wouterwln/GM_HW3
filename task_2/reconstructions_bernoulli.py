from main import load_data
import torch
from modules_bernoulli import *
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
from sklearn.decomposition import PCA

if __name__ == "__main__":
    _, _, test_data = load_data()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_data = DataLoader(test_data, batch_size=1000, shuffle=False)
    test_data = next(iter(test_data))
    x, y = test_data
    x = x.to(device)
    num_var = 1
    num_latent = 16
    num_neurons = [8, 16, 24, 32]
    dropout = 0.2
    maxpool = [0, 1, 2]
    enc = BernoulliEncoder(num_var, num_latent, num_neurons, dropout, maxpool)
    dec = BernoulliDecoder(num_var, num_latent, num_neurons, dropout, maxpool)
    model = BernoulliVAE(enc, dec).to(device)
    model.load_state_dict(torch.load("VAE_bernoulli_16_dimensional"))
    f, axarr = plt.subplots(8, 8)

    with torch.no_grad():
        out = enc(x)
        embeddings = out[1].cpu()
        output = dec(out[0])[1].cpu()
    for i in range(32):
        axarr[i//4,2*(i%4) ].imshow(test_data[0][i].cpu().permute(1, 2, 0).numpy())
        axarr[i//4,2*(i%4) + 1].imshow(output[i].cpu().permute(1, 2, 0).numpy())
    plt.show()