import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from data import load_train_val_test
from train import SensorDataset
from context_model import ContextEmbeddingModel

if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

    train_dataset = SensorDataset(X_train, y_train)
    val_dataset = SensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = 64, shuffle = False)

    model = ContextEmbeddingModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

    best_val_acc = 0.0
    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(y_batch.cpu())
        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        val_acc = accuracy_score(all_labels.cpu().numpy(), all_preds.cpu().numpy())
        
        print(f"Epoch {epoch+1}: val_acc = {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_context_model.pt")

    print(f"Best val accuracy: {best_val_acc:.4f}")