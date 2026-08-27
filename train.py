import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

from data import load_train_val_test
from model import DirectClassifier

class SensorDataset(Dataset):
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            all_preds.append(preds)
            all_labels.append(y_batch)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    return all_preds, all_labels

if __name__ == "__main__":
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    train_dataset = SensorDataset(X_train, y_train)
    val_dataset = SensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 32, shuffle = False)

    model = DirectClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

    best_val_acc = 0.0
    num_epochs = 50

    for epoch in range(num_epochs):
        train_epoch(model, train_loader, criterion, optimizer)
        val_preds, val_labels = evaluate(model, val_loader)
        val_acc = accuracy_score(val_labels.numpy(), val_preds.numpy())
        print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

    print(f"Best val accuracy: {best_val_acc:.4f}")
