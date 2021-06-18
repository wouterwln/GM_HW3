from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from modules import *
import matplotlib.pyplot as plt
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
    train_set = Subset(mnist_trainset, range(40000))
    validation_set = Subset(mnist_trainset, range(40000, 50000))
    test_set = Subset(mnist_trainset, range(50000, 60000))
    return train_set, validation_set, test_set

def train(model, device, train_dataloader, loss_function, optimizer, num_lags=4):
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
            x = batch
            x = x.to(device)
            z, mu_enc, sig_enc = model.encoder()
            z = z.unsqueeze(0)
            x_reconstr, mu_dec, log_sig_dec = model.decoder(z)
            elbo, logp, kl = loss_function(x, mu_dec, log_sig_dec, mu_enc, sig_enc)
            elbo.backward()
            optimizer.step()
            train_epoch_elbo.append(elbo.cpu().item() / x.size()[0])
            train_epoch_logp.append(logp.cpu().item() / x.size()[0])
            train_epoch_kl.append(kl.cpu().item() / x.size()[0])

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
    batch_size = 1
    train_set, validation_set, test_set = load_data()
    single_point = test_set.dataset[0][0]
    single_point_loader = DataLoader(single_point, batch_size=batch_size, shuffle=True)
    num_var = 1
    num_latent = 16
    num_neurons = [8, 16, 24, 32]
    dropout = 0.2
    maxpool = [0, 1, 2]
    enc_trained = EncoderTrained(num_var, num_latent, num_neurons, dropout, maxpool)
    dec = Decoder(num_var, num_latent, num_neurons, dropout, maxpool)
    model = VAE(enc_trained, dec).to(device)
    model.load_state_dict(torch.load("VAE_gaussian_16_dimensional_constant_var"))
    trained_dec = model.decoder
    trained_dec.eval()  # Make sure it doesn't get trained
    enc = Encoder(num_latent)  # Initialize the new encoder
    model = VAE(enc, trained_dec)
    optimizer = optim.Adam(model.parameters())
    model = train(model, device, single_point_loader, ELBOLoss(), optimizer, num_lags=2)
    torch.save(model.state_dict(), "VAE_gaussian_16_task4")
    model.eval()

    f, axarr = plt.subplots(1, 3)
    # Reconstructing
    with torch.no_grad():
        z, mu_enc, log_sig_enc = model.encoder(single_point)
        x_reconstr, mu_dec, log_sig_dec = model.decoder(z)
        axarr[0].imshow(single_point.cpu().permute(1, 2, 0).numpy())  # Real data point
        axarr[1].imshow(mu_dec.cpu().permute(1, 2, 0).numpy())  # Change this to different reconstruction
        axarr[2].imshow(x_reconstr.cpu().permute(1, 2, 0).numpy())  # Reconstruction like task 1D
        plt.show()