import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import warnings
import torch.nn.functional as F
import sys
import math
warnings.filterwarnings('ignore')

DATASETS = ["Baron", "Xin", "Segerstolpe"]
TRAIN_DATA = "Pancreas"
CURRENT_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
NUM_LAYERS = 16
FEATURE_DIR = f"test/{TRAIN_DATA}/classify_data"  
LABEL_DIR = f"test/{TRAIN_DATA}"                  
OUTPUT_DIR = f"results/{TRAIN_DATA}"              
TIMESTEP_VAL = 0.5
LABEL_PATH = f"dataset/{TRAIN_DATA}/{TRAIN_DATA}_labels.npy"
MODEL_PATH = f"Classifier/{TRAIN_DATA}/classify_weights_seed{CURRENT_SEED}_{TIMESTEP_VAL}.pth"
MAPPING_PATH = f"dataset/{TRAIN_DATA}/{TRAIN_DATA}_label_mapping.npy"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INPUT_DIM = 384
NUM_CLASSES = 64  

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

def unify_label_mapping(raw_mapping):
    if isinstance(raw_mapping, np.ndarray):
        if raw_mapping.ndim == 0: raw_mapping = raw_mapping.item()
        else: return {i: str(name) for i, name in enumerate(raw_mapping)}

    if isinstance(raw_mapping, dict):
        first_key = next(iter(raw_mapping))
        first_val = raw_mapping[first_key]
        if isinstance(first_key, str) and first_key.isdigit():
            return {int(k): v for k, v in raw_mapping.items()}
        if isinstance(first_key, str) and isinstance(first_val, (int, np.integer)):
            return {v: k for k, v in raw_mapping.items()}
        return raw_mapping
    
    return {i: f"Cell_{i}" for i in range(NUM_CLASSES)}

def calculate_metrics(y_true, y_pred, y_probs, label_map):
    print("\n📊 Calculating Metrics...")
    # main metric
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0) 
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    # AUROC 
    auroc = 0.0
    try:
        classes_in_test = np.unique(y_true)
        if len(classes_in_test) < 2:
            print(f"  ⚠️ AUROC skipped: Need at least 2 classes, but found {len(classes_in_test)}.")
        else:
            valid_probs = y_probs[:, classes_in_test]
            row_sums = valid_probs.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0 
            valid_probs = valid_probs / row_sums
            auroc = roc_auc_score(
                y_true, 
                valid_probs, 
                multi_class='ovr', 
                average='weighted', 
                labels=classes_in_test
            )
    except Exception as e:
        print(f"  ⚠️ AUROC calculation failed: {e}")

    print("-" * 30)
    print(f"Accuracy  : {acc:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"AUROC     : {auroc:.4f}")
    print("-" * 30)
    
    return {"Acc": acc, "F1": f1, "Prec": precision, "Recall": recall, "AUROC": auroc}

def process_single_dataset(dataset_name, label_map):
    print(f"\n{'='*10} Processing: {dataset_name} (Seed {CURRENT_SEED}) {'='*10}")

    label_path = os.path.join(LABEL_DIR, f"{dataset_name}_labels.npy")
    output_csv = os.path.join(OUTPUT_DIR, f"{dataset_name}_seed{CURRENT_SEED}_results.csv")

    print(f"  Loading {NUM_LAYERS} layers for {dataset_name}...")
    feat_list = []
    for i in range(NUM_LAYERS):
        f_path = os.path.join(FEATURE_DIR, f"{dataset_name}_classify_data_{i}_{TIMESTEP_VAL}.npy")
        
        if not os.path.exists(f_path):
            print(f"❌ Error: Feature file not found: {f_path}")
            return 0.0
            
        feat_list.append(np.load(f_path))

    features_stack = np.stack(feat_list, axis=1)
    features = torch.FloatTensor(features_stack).to(DEVICE)
    
    if os.path.exists(label_path):
        true_labels = np.load(label_path)
    else:
        true_labels = None

    print(f"🚀 Loading model: {MODEL_PATH}")
    model = scDiTA(input_dim=INPUT_DIM, num_classes=NUM_CLASSES, num_layers=NUM_LAYERS).to(DEVICE)  
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model file missing: {MODEL_PATH}")
        return 0.0
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.eval() 
        with torch.no_grad():
            logits = model(features)
            probs = torch.softmax(logits, dim=1).cpu().numpy() 
            final_preds = np.argmax(probs, axis=1)             
            
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return 0.0

    # evaluation and save
    current_acc = 0.0 
    if true_labels is not None:
        true_labels = true_labels.flatten()
        final_preds = final_preds.flatten()

        train_labels_path = LABEL_PATH
        try:
            train_labels = np.load(train_labels_path, allow_pickle=True)
            known_ids = set(train_labels) 
        except:
            print(f"⚠️ Warning: Could not load train labels.")
            known_ids = set(label_map.keys())

        # only calculate the lable in train set 
        id_mask = np.array([label in known_ids for label in true_labels])

        if len(true_labels) != len(final_preds):
             min_len = min(len(true_labels), len(final_preds))
             true_labels, final_preds, probs = true_labels[:min_len], final_preds[:min_len], probs[:min_len]
             id_mask = id_mask[:min_len]

        eval_true = true_labels[id_mask]
        eval_pred = final_preds[id_mask]
        eval_probs = probs[id_mask]
        
        if len(eval_true) > 0:
            metrics = calculate_metrics(eval_true, eval_pred, eval_probs, label_map)
            current_acc = metrics["Acc"]
        else:
            print("❌ No ID samples found after filtering!")
            metrics = {"Acc": 0.0, "F1": 0.0}

    annotated_names = [label_map.get(pred, f"Unknown_{pred}") for pred in final_preds]
    df = pd.DataFrame({
        "Cell_Index": range(len(final_preds)),
        "Predicted_Label_ID": final_preds,
        "Predicted_Cell_Type": annotated_names
    })
    
    df.to_csv(output_csv, index=False)
    print(f"✅ Results saved to: {output_csv}")
    
    return [metrics["Acc"], metrics["F1"]] if true_labels is not None else [0.0, 0.0]

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    print(f"Device: {DEVICE}")
    try:
        raw_map = np.load(MAPPING_PATH, allow_pickle=True)
        label_map = unify_label_mapping(raw_map)
    except:
        label_map = {}

    seed_results = {}

    for dataset in DATASETS:
        res = process_single_dataset(dataset, label_map)
        seed_results[dataset] = res  

if __name__ == "__main__":
    main()