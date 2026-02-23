def train(model, data_loader, optimizer, loss_fn):
    model.train()
    for x, y in data_loader:
        optimizer.zero_grad()
        preds = model(x)
        loss = loss_fn(preds, y)
        loss.backward()
        optimizer.step()
