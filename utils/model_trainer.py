import tqdm
import torch


def autoencoder_trainer(model, criteria, optimizer, train_loader,
                        test_loader, device, epochs):
    train_loss_hist = []
    test_loss_hist = []
    
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = []
        test_loss = []
        for x, _ in train_loader:
            x = x.to(device)
            model.train()
            _, reconstruct = model(x)
            loss = criteria(reconstruct, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss_hist.append(torch.tensor(train_loss).mean())
        model.eval()
        for x, _ in test_loader:
            x = x.to(device)
            _, reconstruct = model(x)
            loss = criteria(reconstruct, x)
            test_loss.append(loss.item())
        train_loss_hist.append(torch.tensor(test_loss).mean())
        if epoch % 10 == 0:    
            print(f'epochs: {epoch + 1} - Train loss: {train_loss[-1]} - Test loss: {test_loss[-1]}')
    return train_loss_hist, test_loss_hist


def vae_trainer(model, criteria, optimizer, train_loader,
                test_loader, device, epochs, beta=1):
    
    train_loss_hist = []
    test_loss_hist = []
    
    for epoch in tqdm.tqdm(range(epochs)):
        train_loss = []
        test_loss = []
        for x, _ in train_loader:
            x = x.to(device)
            model.train()
            mu, logvar, reconstruct = model(x)
            reconstruction_loss, kl_loss = criteria(reconstruct, x, mu, logvar)
            reconstruction_loss = reconstruction_loss / x.shape[0]
            kl_loss = kl_loss / x.shape[0]
            loss = reconstruction_loss + beta * kl_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        train_loss_hist.append(torch.tensor(train_loss).mean())
        
        model.eval()
        for x, _ in test_loader:
            x = x.to(device)
            mu, logvar, reconstruct = model(x)
            reconstruction_loss, kl_loss = criteria(reconstruct, x, mu, logvar)
            reconstruction_loss = reconstruction_loss / x.shape[0]
            kl_loss = kl_loss / x.shape[0]
            loss = reconstruction_loss + beta * kl_loss
            test_loss.append(loss.item())
        test_loss_hist.append(torch.tensor(test_loss).mean())
        
        if epoch % 10 == 0:     
            print(f'epochs: {epoch + 1} - Train loss: {train_loss[-1]} - Test loss: {test_loss[-1]}')
            print(f'Test Error Ratio || reconstruction error: {reconstruction_loss} , KLD: {kl_loss}')
    return train_loss_hist, test_loss_hist
