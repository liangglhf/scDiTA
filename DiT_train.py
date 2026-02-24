import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import numpy as np
from tqdm import tqdm
import os 
from model import DiT
import copy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1024
LR = 1e-3
EPOCHS = 5000
START_EPOCH = 0 
PLATEAU_EPOCHS = 1500     
DECAY_EPOCHS = EPOCHS - PLATEAU_EPOCHS
DATASET = "Pancreas"

class LoadDataset(Dataset):
    def __init__(self, data, labels_path="labels.npy"):
        try:
            if isinstance(data, str):
                data_np = np.load(data)
            else:
                data_np = data
                
            self.data = torch.from_numpy(data_np).float()
            self.labels = torch.from_numpy(np.load(labels_path)).long()
        except Exception as e:
            print(f"Error: fail to load data - {e}")
            raise

        self.num_classes = len(torch.unique(self.labels))
        print(f"Loaded dataset with {len(self.data)} cells, {self.num_classes} classes.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            "latents": self.data[idx],       # (64, 16)
            "class_labels": self.labels[idx] # (1,)
        }

def preprocess_data(data_path):
    data = np.load(data_path)
    vmin, vmax = np.percentile(data, [0.1, 99.9])
    data = np.clip(data, vmin, vmax)
    mean = data.mean()
    std = data.std()
    data_norm = (data - mean) / std
    np.save(f"dataset/{DATASET}/{DATASET}_norm.npy", {"mean": mean, "std": std})
    return data_norm

def update_ema(ema_model, model, decay):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def train():
    latents_norm = preprocess_data(f"dataset/{DATASET}/{DATASET}_latent_tokens.npy")

    dataset = LoadDataset(
        data=latents_norm, 
        labels_path=f"dataset/{DATASET}/{DATASET}_labels.npy"
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
 
    model = DiT(
        input_dim=16,
        hidden_dim=384,   
        num_layers=16,   
        num_heads=6,      
        num_classes=64    
    ).to(DEVICE)

    ema_model = copy.deepcopy(model).to(DEVICE) 
    ema_decay = 0.999

    # start in mid checkpoint
    #checkpoint_path = f"DiT/{DATASET}/{DATASET}.pt"
    #model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    #print(f"✅ loaded checkpoint: {checkpoint_path}")

    optimizer = AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DECAY_EPOCHS, eta_min=1e-5)
    model.train()
    print("Training Start...")

    os.makedirs(f"DiT/{DATASET}", exist_ok=True)

    for epoch in range(START_EPOCH, EPOCHS):
        avg_loss = 0 
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False) 
        
        for batch in pbar:
            x_1 = batch["latents"].to(DEVICE, non_blocking=True) 
            labels = batch["class_labels"].to(DEVICE, non_blocking=True)
            bsz = x_1.shape[0]
            # 1. sampling nosie
            x_0 = torch.randn_like(x_1).to(DEVICE)
            # 2. sampling timestep
            t = torch.rand((bsz,), device=DEVICE)
            # 3. linear interpolating
            t_broad = t.view(-1, 1, 1)
            x_t = (1 - t_broad) * x_0 + t_broad * x_1
            # 4. velocity
            v_target = x_1 - x_0
            # 5. CFG Masking
            mask = torch.rand(bsz, device=DEVICE) < 0.1
            labels[mask] = model.num_classes
            # 6. Loss
            v_pred = model(x_t, t * 1000, labels)
            loss = F.mse_loss(v_pred, v_target)
            # back propagation
            optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            update_ema(ema_model, model, ema_decay)
            avg_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss_val = avg_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        log_msg = f"Epoch {epoch+1}/{EPOCHS} | Avg Loss: {avg_loss_val:.6f} | LR: {current_lr:.8f}"
        print(log_msg)

        # update lr
        if epoch >= PLATEAU_EPOCHS:
            scheduler.step()

        # save model
        if (epoch + 1) % 200 == 0:
            torch.save(model.state_dict(), f"DiT/{DATASET}/{DATASET}_epoch_{epoch+1}.pt")
            torch.save(ema_model.state_dict(), f"DiT/{DATASET}/{DATASET}_ema_epoch_{epoch+1}.pt") # use this in extract.py

    final_epoch = EPOCHS
    torch.save(model.state_dict(), f"DiT/{DATASET}/{DATASET}_epoch_{final_epoch}.pt")
    torch.save(ema_model.state_dict(), f"DiT/{DATASET}/{DATASET}_ema_epoch_{final_epoch}.pt") # use this in extract.py
    print(f"🎉 Traning Finished, the model is saved to epoch {final_epoch}")

if __name__ == "__main__":
    train()