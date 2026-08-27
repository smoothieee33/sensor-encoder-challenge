import torch
import torch.nn as nn

class SensorEncoder(nn.Module):
    def __init__(self, embed_dim = 64):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=9, out_channels = 16, kernel_size = 5, padding = 2)
        self.pool = nn.MaxPool1d(kernel_size = 2)
        self.conv2 = nn.Conv1d(in_channels= 16, out_channels = 32, kernel_size = 5, padding = 2)
        self.conv3 = nn.Conv1d(in_channels= 32, out_channels = embed_dim, kernel_size = 5, padding = 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.mean(dim=2)
        return x

class DirectClassifier(nn.Module):
    def __init__(self, embed_dim = 64, num_classes=6):
        super().__init__()
        self.encoder = SensorEncoder(embed_dim) 
        self.head = nn.Linear(in_features= embed_dim, out_features=num_classes)

    def forward(self, x):
        next = self.encoder(x)
        logits = self.head(next)
        return logits

if __name__ == "__main__":
    model = DirectClassifier()
    fake_run = torch.randn(8, 128, 9)
    out = model(fake_run)
    print("Run", out.shape)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("train those dawgs:", num_params)
