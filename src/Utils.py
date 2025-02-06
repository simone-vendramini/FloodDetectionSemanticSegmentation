import torch


def train(dataloader, model, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for _, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        y = y.squeeze()

        # Compute prediction error
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        predicted = pred.argmax(dim=1)
        total += y.numel()
        correct += (predicted == y).sum().item()

    return running_loss / len(dataloader), correct / total


def validate(dataloader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            y = y.squeeze()
            # Compute prediction error
            pred = model(X)
            loss = criterion(pred, y)

            running_loss += loss.item()
            predicted = pred.argmax(dim=1)
            total += y.numel()
            correct += (predicted == y).sum().item()

    return running_loss / len(dataloader), correct / total

def train_finetuning(train_loader, model, criterion, optimizer, device):
    # Training phase
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_pixels = 0

    for inputs, masks in train_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(outputs, dim=1)
        running_corrects += torch.sum(preds == masks).item()
        total_pixels += masks.numel()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = running_corrects / total_pixels
    print(f'Training Loss: {epoch_loss:.4f} | Training Acc: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc
        
def validate_finetuning(val_loader, model, criterion, device):
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    val_pixels = 0

    with torch.no_grad():
        for inputs, masks in val_loader:
            inputs, masks = inputs.to(device), masks.to(device)
            outputs = model(inputs)['out']
            loss = criterion(outputs, masks)
            val_loss += loss.item() * inputs.size(0)

            preds = torch.argmax(outputs, dim=1)
            val_corrects += torch.sum(preds == masks).item()
            val_pixels += masks.numel()

    val_loss /= len(val_loader.dataset)
    val_acc = val_corrects / val_pixels
    print(f'Validation Loss: {val_loss:.4f} | Validation Acc: {val_acc:.4f}')
    return val_loss, val_acc