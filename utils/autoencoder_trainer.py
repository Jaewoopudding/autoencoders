def train(model, criteria, optimizer, train_loader,
          test_loader, device, epochs):
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        for x, _ in train_loader:
            x = x.to(device)
            model.train()
            _, reconstruct = model(x)
            loss = criteria(reconstruct, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        
        model.eval()
        for x, _ in test_loader:
            x = x.to(device)
            _, reconstruct = model(x)
            loss = criteria(reconstruct, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            test_loss.append(loss.item())
            
        print(f'epochs: {epoch + 1} - Train error: {train_loss[-1]} - Test error: {test_loss[-1]}')
    return train_loss, test_loss