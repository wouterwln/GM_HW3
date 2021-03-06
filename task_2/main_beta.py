from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from modules_beta import *
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_set = Subset(mnist_trainset, range(40000))
    validation_set = Subset(mnist_trainset, range(40000, 50000))
    test_set = Subset(mnist_trainset, range(50000, 60000))
    return train_set, validation_set, test_set

def train(model, device, train_dataloader, validation_dataloader, loss_function, optimizer, num_lags=4):
    training = True
    lag_valloss = [np.inf for _ in range(num_lags)]
    train_loss = []
    val_loss = []
    epoch = 1
    while training:
        train_epoch_elbo, train_epoch_logp, train_epoch_kl = [], [], []
        val_epoch_elbo, val_epoch_logp, val_epoch_kl = [], [], []
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)
            z, mu_enc, log_sig_enc = model.encoder(x)
            x_reconstr, alpha_dec, beta_dec = model.decoder(z)
            elbo, logp, kl = loss_function(x, torch.exp(alpha_dec), torch.exp(beta_dec), mu_enc, log_sig_enc)
            elbo.backward()
            optimizer.step()
            train_epoch_elbo.append(elbo.cpu().item() / x.size()[0])
            train_epoch_logp.append(logp.cpu().item() / x.size()[0])
            train_epoch_kl.append(kl.cpu().item() / x.size()[0])
        model.eval()
        with torch.no_grad():
            for batch in validation_dataloader:
                x, _ = batch
                x = x.to(device)
                z, mu_enc, log_sig_enc = model.encoder(x)
                x_reconstr, alpha_dec, beta_dec = model.decoder(z)
                elbo, logp, kl = loss_function(x, torch.exp(alpha_dec), torch.exp(beta_dec), mu_enc, log_sig_enc)
                val_epoch_elbo.append(elbo.cpu().item() / x.size()[0])
                val_epoch_logp.append(logp.cpu().item() / x.size()[0])
                val_epoch_kl.append(kl.cpu().item() / x.size()[0])
            if epoch % 5 == 0:
                plt.figure()

                # subplot(r,c) provide the no. of rows and columns
                f, axarr = plt.subplots(7, 3)

                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                for i in range(7):
                    axarr[i, 0].imshow(x[i].cpu().permute(1, 2, 0).numpy())
                    # axarr[i, 1].imshow(mu_dec[i].cpu().permute(1, 2, 0).numpy())
                    axarr[i, 2].imshow(x_reconstr[i].cpu().permute(1, 2, 0).numpy())
                plt.show()

        print("Epoch loss in epoch {}: {}, logP(X): {}, KL Divergence: {}".format(epoch, np.mean(np.array(train_epoch_elbo)), np.mean(np.array(train_epoch_logp)), np.mean(np.array(train_epoch_kl))))
        print("Epoch validation loss in epoch {}: {}, logP(X): {}, KL Divergence: {}".format(epoch, np.mean(np.array(val_epoch_elbo)), np.mean(np.array(val_epoch_logp)), np.mean(np.array(val_epoch_kl))))
        lag_valloss.append(np.mean(np.array(val_epoch_elbo)))
        train_loss.append(-np.mean(np.array(train_epoch_elbo)))
        val_loss.append(-np.mean(np.array(val_epoch_elbo)))
        if sorted(lag_valloss) == lag_valloss:
            training = False
        else:
            lag_valloss = lag_valloss[1:]
            epoch += 1
    plt.figure()
    plt.plot(train_loss, label="Train")
    plt.plot(val_loss, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO")
    plt.legend()
    plt.show()
    return model

if __name__ == '__main__':
    batch_size = 1000
    train_set, validation_set, test_set = load_data()
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_set = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    num_var = 1
    # TODO: Run 2 and 16
    num_latent = 16
    num_neurons = [8, 16, 24, 32]
    dropout = 0.2
    maxpool = [0, 1, 2]
    enc = BetaEncoder(num_var, num_latent, num_neurons, dropout, maxpool)
    dec = BetaDecoder(num_var, num_latent, num_neurons, dropout, maxpool)
    model = BetaVAE(enc, dec).to(device)
    optimizer = optim.Adam(model.parameters())
    model = train(model, device, train_set, test_set, BetaELBOLoss(), optimizer, num_lags=3)
    torch.save(model.state_dict(), "VAE_beta_16_dimensional")
    model.eval()
    with torch.no_grad():
        sample = model.sample(16)
        f, axarr = plt.subplots(4, 4)
        for i in range(16):

            axarr[i // 4, i % 4].imshow(sample[i].cpu().permute(1, 2, 0).numpy())
        plt.show()