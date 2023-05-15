import matplotlib.pyplot as plt


def visualization(loader, model, device, num_of_samples=5):
    for x, _ in loader:
        recon = model(x.to(device))[-1]
        fig, ax = plt.subplots(num_of_samples, 2, figsize=(4, 10))
        plt.suptitle("Left : Original, Right : Reconstructed")
        for i in range(num_of_samples):
            ax[i, 0].imshow(x[i].view(28, 28).detach().numpy())
            ax[i, 1].imshow(recon[i].cpu().view(28, 28).detach().numpy())
        plt.show()
        break
    