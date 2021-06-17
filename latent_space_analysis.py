from main import load_data
import torch
from modules import Encoder, Decoder, VAE
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
    num_latent = 2
    num_neurons = [8, 16, 24, 32]
    dropout = 0.2
    maxpool = [0, 1, 2]
    enc = Encoder(num_var, num_latent, num_neurons, dropout, maxpool)
    dec = Decoder(num_var, num_latent, num_neurons, dropout, maxpool)
    model = VAE(enc, dec).to(device)
    model.load_state_dict(torch.load("VAE_gaussian_2_dimensional_constant_var"))
    f, axarr = plt.subplots(8, 8)

    with torch.no_grad():
        out = enc(x)
        embeddings = out[1].cpu()
        output = dec(out[0])[0].cpu()
    for i in range(32):
        axarr[i//4,2*(i%4) ].imshow(test_data[0][i].cpu().permute(1, 2, 0).numpy())
        axarr[i//4,2*(i%4) + 1].imshow(output[i].cpu().permute(1, 2, 0).numpy())
    plt.show()
    if embeddings.shape[1] > 2:
        pca = PCA(n_components=2)
        embeddings = embeddings.numpy()
        embeddings = pca.fit_transform(embeddings)
    colors = cm.rainbow(np.linspace(0, 1, 10))
    plt.figure()
    for y_v, c in enumerate(colors):
        data = embeddings[y == y_v]
        plt.scatter(data[:,0], data[:,1], color = c)
    plt.show()

    x = torch.cat([x[y == 1][0].unsqueeze(0), x[y == 8][0].unsqueeze(0)])
    with torch.no_grad():
        embeddings = enc(x)[1].cpu()
    f, axarr = plt.subplots(1, 12)

    embeddings = torch.cat([(k * embeddings[1] + (1-k) * embeddings[0]).unsqueeze(0) for k in np.linspace(0, 1, 10)]).to(device)
    with torch.no_grad():
        images = dec(embeddings)[1].cpu()
    axarr[0].imshow(x[0].cpu().permute(1, 2, 0).numpy())
    axarr[11].imshow(x[1].cpu().permute(1, 2, 0).numpy())
    for i in range(10):
        axarr[i+1].imshow(images[i].permute(1, 2, 0).numpy())
    plt.show()




