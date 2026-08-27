import torch
from sklearn.metrics import f1_score

from data import load_train_val_test
from model import DirectClassifier
from train import SensorDataset
from train import evaluate
from torch.utils.data import DataLoader

(X_train, y_train), (X_val, y_val), (X_test, y_test) = load_train_val_test()

test_dataset = SensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False)

model = DirectClassifier()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

test_preds, test_labels = evaluate(model, test_loader)
macro_f1 = f1_score(test_labels.numpy(), test_preds.numpy(), average = 'macro')
print(f"Direct classifier test macro-F1: {macro_f1:.4f}")