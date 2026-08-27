import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from data import load_train_val_test
from train import SensorDataset
from context_model import ContextEmbeddingModel

torch.manual_seed(67) #im just a kidddddddd

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

test_dataset = SensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

model = ContextEmbeddingModel().to(device)
model.load_state_dict(torch.load("best_context_model.pt"))
model.eval()

all_embeddings = []
all_labels = []
batch_size = 32
all_preds = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        emb = model.get_embedding(X_batch)
        all_embeddings.append(emb)
        all_labels.append(y_batch)

all_embeddings = torch.cat(all_embeddings)
all_labels = torch.cat(all_labels)

num_test_samples = all_embeddings.shape[0]
perm = torch.randperm(num_test_samples)
shuffled = all_embeddings[perm]

with torch.no_grad():
    for i in range(0, num_test_samples, batch_size):
        bitch = shuffled[i:i+batch_size]
        logits = model.classify_from_embedding(bitch)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())

all_preds = torch.cat(all_preds) 
macro_f1 = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average = 'macro')
print(f"Context Model test macro-F1: {macro_f1:.4f}")