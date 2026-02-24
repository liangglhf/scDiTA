import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import sys
import math

DATA_NAME = "Pancreas"
NUM_LAYERS = 16
LABEL_PATH = f"dataset/{DATA_NAME}/{DATA_NAME}_labels.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
LEARNING_RATE = 0.1
EPOCHS = 400
NUM_CLASSES = 64
INPUT_DIM = 384
TIMESTEP_VAL = 0.5

class scDiTA(nn.Module):
    def __init__(self, input_dim, num_classes, num_layers=16):
        super().__init__()
        self.lora_adapters = nn.ModuleList([LoRA(input_dim, rank=8) for _ in range(num_layers)])

        init_weights = torch.zeros(num_layers)
        init_weights[int(num_layers * 0.5):int(num_layers * 0.8)] = 3.0  # giving the layers [8, 12] initial weights
        self.layer_weights = nn.Parameter(init_weights)

        hidden_dim = 2048
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU()         
        self.dropout = nn.Dropout(p=0.5) 
        self.layer2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        corrected_features_list = []
        for i in range(len(self.lora_adapters)):
            feat_i = x[:, i, :] 
            feat_i_new = self.lora_adapters[i](feat_i)
            corrected_features_list.append(feat_i_new)
        x = torch.stack(corrected_features_list, dim=1)

        weights = F.softmax(self.layer_weights, dim=0)
        if self.training:
            weights = F.dropout(weights, p=0.4, training=True)
            if weights.sum() > 0:
                weights = weights / (weights.sum() + 1e-9)
            else:
                weights = torch.ones_like(weights) / len(weights)

        x = torch.sum(x * weights.view(1, -1, 1), dim=1) 
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
    
class LoRA(nn.Module):
    def __init__(self, input_dim, rank=8, lora_alpha=8, lora_dropout=0.2): 
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank
        # A (384 -> 16)
        self.lora_A = nn.Linear(input_dim, rank, bias=False)
        # B (16 -> 384)
        self.lora_B = nn.Linear(rank, input_dim, bias=False)
        self.dropout = nn.Dropout(p=lora_dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return x + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        
def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def main():
    print(f"Device: {DEVICE}")
    current_seed = int(sys.argv[1]) if len(sys.argv) > 1 else 42
    torch.manual_seed(current_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(current_seed)

    # 1. load data
    print("1. Loading training data...")
    labels = np.load(LABEL_PATH)
    labels = torch.LongTensor(labels).to(DEVICE)

    all_features = []
    for i in range(NUM_LAYERS):
        f_path = f"dataset/{DATA_NAME}/classify_data/{DATA_NAME}_classify_data_{i}_{TIMESTEP_VAL}.npy"
        feat = np.load(f_path) 
        all_features.append(feat)
    features_stack = np.stack(all_features, axis=1) 
    features = torch.FloatTensor(features_stack).to(DEVICE)

    # 2. DataLoader
    dataset = TensorDataset(features, labels)
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. init model
    model = scDiTA(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, num_layers=NUM_LAYERS).to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4, nesterov=True) 
    #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6) # warmup mode
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. traning loop
    print(f"\n2. Start Training...")
    model.train()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if model.training:
                noise = torch.randn_like(x) * 0.05  
                x = x + noise
            
            optimizer.zero_grad()
            inputs, targets_a, targets_b, lam = mixup_data(x, y, alpha=1.0, use_cuda=(DEVICE=="cuda"))
            inputs, targets_a, targets_b = map(torch.autograd.Variable, (inputs, targets_a, targets_b))
            logits = model(inputs)
            loss = mixup_criterion(criterion, logits, targets_a, targets_b, lam)

            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        avg_loss = total_loss / len(train_loader)
        acc = 100 * correct / total
        
        if (epoch + 1) % 10 == 0:
            print(f"   Epoch {epoch+1:03d}/{EPOCHS} | LR: {current_lr:.6f} | Loss: {avg_loss:.4f} | Acc: {acc:.2f}%")

    # 5. save model
    print("\n3. Saving model weights...")
    torch.save(model.state_dict(), f"Classifier/{DATA_NAME}/classify_weights_seed{current_seed}_{TIMESTEP_VAL}.pth")
    print(f"✅ Model weights saved to: {f'Classifier/{DATA_NAME}/classify_weights_seed{current_seed}_{TIMESTEP_VAL}.pth'}")

if __name__ == "__main__":
    main()