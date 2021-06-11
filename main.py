import matplotlib.pyplot as plt
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from modules import *
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_set = Subset(mnist_trainset, range(50000))
    test_set = Subset(mnist_trainset, range(50000, 60000))
    return train_set, test_set

def train(model, device, train_dataloader, test_dataloader, num_epochs, loss_function, optimizer):

    for epoch in range(num_epochs):
        epoch_elbo = []
        epoch_logp = []
        epoch_kl = []
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)
            z, mu_enc, log_sig_enc = model.encoder(x)
            x_reconstr, mu_dec, log_sig_dec = model.decoder(z)
            elbo, logp, kl = loss_function(x, mu_dec, log_sig_dec, mu_enc, log_sig_enc)
            elbo.backward()
            optimizer.step()
            epoch_elbo.append(elbo.cpu().item() / x.size()[0])
            epoch_logp.append(logp.cpu().item() / x.size()[0])
            epoch_kl.append(kl.cpu().item() / x.size()[0])

        if epoch % 1 == 0:
            with torch.no_grad():
                plt.figure()

                # subplot(r,c) provide the no. of rows and columns
                f, axarr = plt.subplots(7, 3)

                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                for i in range(7):
                    axarr[i, 0].imshow(x[i].cpu().permute(1, 2, 0).numpy())
                    axarr[i, 1].imshow(mu_dec[i].cpu().permute(1, 2, 0).numpy())
                    axarr[i, 2].imshow(x_reconstr[i].cpu().permute(1, 2, 0).numpy())
                plt.show()

        print("Epoch loss in epoch {}: {}, logP(X): {}, KL Divergence: {}".format(epoch, np.mean(np.array(epoch_elbo)), np.mean(np.array(epoch_logp)), np.mean(np.array(epoch_kl))))
    return model

if __name__ == '__main__':
    batch_size = 1000
    train_set, test_set = load_data()
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    num_var = 1
    num_latent = 16
    num_neurons = [8, 16, 24, 32]
    dropout = 0.2
    maxpool = [0, 1, 2]
    enc = Encoder(num_var, num_latent, num_neurons, dropout, maxpool, flatten=True)
    dec = Decoder(num_var, num_latent, num_neurons, dropout, maxpool, flatten=True)
    model = VAE(enc, dec).to(device)
    optimizer = optim.Adam(model.parameters())
    model = train(model, device, train_set, test_set, 50, ELBOLoss(), optimizer)
    model.eval()
    with torch.no_grad():
        sample = model.sample(10)
        for i in range(10):
            plt.figure()
            plt.imshow(sample[i].cpu().permute(1, 2, 0).numpy())
            plt.show()
