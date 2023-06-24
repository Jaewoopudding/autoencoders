import matplotlib.pyplot as plt


def visualization(loader, model, device, num_of_samples=5):
    for x, _ in loader:
        recon = model(x.to(device))[-1]
        fig, ax = plt.subplots(2, num_of_samples, figsize=(10, 4))
        plt.suptitle("Up : Original, Down : Reconstructed")
        for i in range(num_of_samples):
            ax[0, i].imshow(x[i].view(28, 28).detach().numpy(), cmap='gray')
            ax[1, i].imshow(recon[i].cpu().view(28, 28).detach().numpy(), cmap='gray')
        plt.show()
        break
    