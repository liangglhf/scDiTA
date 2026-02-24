import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from tqdm import tqdm
from model import DiT
import sys

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET = "Pancreas"
#CHECKPOINT_PATH = f"DiT/{DATASET}/{DATASET}_epoch_5000.pt" # this is the right way
CHECKPOINT_PATH = f"DiT/{DATASET}/{DATASET}.pt" # this just used in the demo

if len(sys.argv) > 1 and sys.argv[1] == "train":
    INPUT_DIR = f"dataset/{DATASET}"
    OUTPUT_DIR = f"dataset/{DATASET}/classify_data"
    TARGET_DATASETS = [f"{DATASET}"]
else:
    INPUT_DIR = f"test/{DATASET}"
    OUTPUT_DIR = f"test/{DATASET}/classify_data"
    TARGET_DATASETS = ["Baron", "Xin", "Segerstolpe"]

NORM_DATA = f"dataset/{DATASET}/{DATASET}_norm.npy"
MODEL_CONFIG = {
    "input_dim": 16,
    "num_layers": 16,
    "hidden_dim": 384,
    "num_heads": 6,
    "seq_len": 64,
    "num_classes": 64
}
BATCH_SIZE = 128
TIMESTEP_VAL = 0.50

# ==========================================
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
    
def extract_features_for_dataset(model, dataset_name):
    print(f"\n{'='*20} Processing: {dataset_name} {'='*20}")
    data_path = os.path.join(INPUT_DIR, f"{dataset_name}_latent_tokens.npy")
    label_path = os.path.join(INPUT_DIR, f"{dataset_name}_labels.npy")
    norm_path = NORM_DATA
    raw_data = np.load(data_path, allow_pickle=True)
    norm_params = np.load(norm_path, allow_pickle=True).item()
    vmin, vmax = np.percentile(raw_data, [0.1, 99.9])
    raw_data = np.clip(raw_data, vmin, vmax)
    normalized_data = (raw_data - norm_params['mean']) / norm_params['std']
    dataset = LoadDataset(data=normalized_data, labels_path=label_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    for layer_idx in range(MODEL_CONFIG["num_layers"]):
        print(f"\n>>> Extracting Layer {layer_idx}/{MODEL_CONFIG['num_layers']-1}")
        current_layer_features = []
        current_batch_inputs = {}
        current_batch_params = {}
        def get_inputs_hook(module, args):
            current_batch_inputs["z"] = args[0].detach()
        def get_params_hook(module, input, output):
            current_batch_params["ada_params"] = output.detach()

        h_layer = model.blocks[layer_idx].register_forward_pre_hook(get_inputs_hook)
        h_ada = model.adaLN_modulation.register_forward_hook(get_params_hook)

        model.eval()
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Layer {layer_idx} Progress"):
                latents = batch["latents"].to(DEVICE)
                bsz = latents.shape[0]
                
                # input clean data
                t_input = torch.full((bsz,), TIMESTEP_VAL * 1000, device=DEVICE).float()
                c_input = torch.full((bsz,), model.num_classes, device=DEVICE).long()
                _ = model(latents, timestep=t_input, class_labels=c_input)

                # Post-AdaLN parameter
                ada_params = current_batch_params["ada_params"]
                chunks = ada_params.chunk(6, dim=1)
                beta, gamma = chunks[0].unsqueeze(1), chunks[1].unsqueeze(1)
                z = current_batch_inputs["z"]
                mu, var = z.mean(dim=-1, keepdim=True), z.var(dim=-1, keepdim=True, unbiased=False)
                z_hat = (1 + gamma) * ((z - mu) / torch.sqrt(var + 1e-5)) + beta
                current_layer_features.append(z_hat.mean(dim=1).cpu())
                
                # We flow the method of "Unleashing Diffusion Transformers for Visual" 
                batch_max = z_hat.abs().max(dim=1)[0].max(dim=0)[0].cpu()
                if 'layer_max_val' not in locals(): layer_max_val = batch_max
                else: layer_max_val = torch.max(layer_max_val, batch_max)

                current_batch_inputs.clear()
                current_batch_params.clear()

        h_layer.remove()
        h_ada.remove()

        # fearture 
        features_pooled = torch.cat(current_layer_features, dim=0).numpy()
        del current_layer_features

        # medain and threshold
        global_median = np.median(np.abs(features_pooled))
        threshold = global_median * 100.0
        
        # massive channel discard
        outlier_indices = np.where(layer_max_val.numpy() > threshold)[0]
        if len(outlier_indices) > 0:
            print(f"🚨 Layer {layer_idx}: DISCARDED {len(outlier_indices)} massive channels!")
            features_pooled[:, outlier_indices] = 0.0
        else:
            print(f"✅ Layer {layer_idx}: No massive activations.")
        
        if 'layer_max_val' in locals(): del layer_max_val

        # BatchNorm and Save
        bn_layer = nn.BatchNorm1d(MODEL_CONFIG["hidden_dim"], affine=False)
        with torch.no_grad():
            features_final = bn_layer(torch.from_numpy(features_pooled)).numpy()
        save_filename = f"{dataset_name}_classify_data_{layer_idx}_{TIMESTEP_VAL}.npy"
        save_path = os.path.join(OUTPUT_DIR, save_filename)
        np.save(save_path, features_final)
        print(f"Saved: {save_path} | Shape: {features_final.shape}")
        torch.cuda.empty_cache()

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR) 
    print(f"Device: {DEVICE}")
    print(f"Mode: Full Layer Extraction (0 to {MODEL_CONFIG['num_layers']-1})")
    print("\nLoading Model...")
    model = DiT(**MODEL_CONFIG).to(DEVICE)
    try:
        state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict, strict=False)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading weights: {e}")
        return
    for dataset_name in TARGET_DATASETS:
        extract_features_for_dataset(model, dataset_name)
    print("\n" + "="*50)
    print("All datasets processed.")

if __name__ == "__main__":
    main()