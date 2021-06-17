from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from modules_bernoulli import *
from torch.nn.functional import relu
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
    epoch = 1
    while training:
        train_epoch_obj, train_epoch_cross_entropy, train_epoch_kl = [], [], []
        val_epoch_obj, val_epoch_cross_entropy, val_epoch_kl = [], [], []
        model.train()
        for batch in train_dataloader:
            optimizer.zero_grad()
            x, _ = batch
            x = x.to(device)
            z, mu_enc, log_sig_enc = model.encoder(x)
            x_reconstr, param = model.decoder(z)

            obj, cross_entropy, kl = loss_function(x, param, mu_enc, log_sig_enc)
            obj.backward()
            optimizer.step()
            train_epoch_obj.append(obj.cpu().item() / x.size()[0])
            train_epoch_cross_entropy.append(cross_entropy.cpu().item() / x.size()[0])
            train_epoch_kl.append(kl.cpu().item() / x.size()[0])
        model.eval()
        with torch.no_grad():
            for batch in validation_dataloader:
                x, _ = batch
                x = x.to(device)
                z, mu_enc, log_sig_enc = model.encoder(x)
                x_reconstr, param = model.decoder(z)

                obj, cross_entropy, kl = loss_function(x, param, mu_enc, log_sig_enc)
                val_epoch_obj.append(obj.cpu().item() / x.size()[0])
                val_epoch_cross_entropy.append(cross_entropy.cpu().item() / x.size()[0])
                val_epoch_kl.append(kl.cpu().item() / x.size()[0])
            if epoch % 5 == 0:
                plt.figure()

                # subplot(r,c) provide the no. of rows and columns
                f, axarr = plt.subplots(7, 3)

                # use the created array to output your multiple images. In this case I have stacked 4 images vertically
                for i in range(7):
                    axarr[i, 0].imshow(x[i].cpu().permute(1, 2, 0).numpy())
                    axarr[i, 1].imshow(param[i].cpu().permute(1, 2, 0).numpy())
                    axarr[i, 2].imshow(x_reconstr[i].permute(1, 2, 0).cpu().numpy())

                plt.show()

        print("Objective loss in epoch {}: {}, cross_entropy(X): {}, KL Divergence: {}".format(epoch, np.mean(np.array(train_epoch_obj)), np.mean(np.array(train_epoch_cross_entropy)), np.mean(np.array(train_epoch_kl))))
        print("Objective validation loss in epoch {}: {}, cross_entropy(X): {}, KL Divergence: {}".format(epoch, np.mean(np.array(val_epoch_obj)), np.mean(np.array(val_epoch_cross_entropy)), np.mean(np.array(val_epoch_kl))))
        lag_valloss.append(np.mean(np.array(val_epoch_obj)))
        if sorted(lag_valloss) == lag_valloss:
            training = False
        else:
            lag_valloss = lag_valloss[1:]
            epoch += 1
    return model

if __name__ == '__main__':
    batch_size = 1000
    train_set, validation_set, test_set = load_data()
    train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_set = DataLoader(validation_set, batch_size=batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    num_var_enc = 1
    num_var_dec = 1
    # TODO: Run 2 and 16
    num_latent = 16
    num_neurons = [8, 16, 24, 32]
    dropout = 0.2
    maxpool = [0, 1, 2]
    enc = BernoulliEncoder(num_var_enc, num_latent, num_neurons, dropout, maxpool)
    dec = BernoulliDecoder(num_var_dec, num_latent, num_neurons, dropout, maxpool)
    model = BernoulliVAE(enc, dec).to(device)
    loss_function = BernoulliELBOLoss()
    optimizer = optim.Adam(model.parameters())
    model = train(model, device, train_set, test_set, BernoulliELBOLoss(), optimizer, num_lags=3)
    torch.save(model.state_dict(), "Bernoulli_beta_2_dimensional")
    model.eval()
    with torch.no_grad():
        sample = model.sample(10)
        for i in range(10):
            plt.figure()
            plt.imshow(sample[i].cpu().permute(1, 2, 0).numpy())
            plt.show()
