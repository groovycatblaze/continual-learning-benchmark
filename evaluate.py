def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for x, y in data_loader:
            preds = model(x)
            predicted = preds.argmax(dim=1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
    
    return correct / total
