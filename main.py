import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import copy
import os
import argparse
import pickle

from src.utils import set_seed
from src.models import SimpleCNN
from src.data_loader import get_dataset_realistic_noniid
from src.client import Client

# ==========================================
# Configuration
# ==========================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='CIFAR10') # 기본값 고정
args = parser.parse_args()

DATASET_NAME = 'CIFAR10' # 강제 고정
NUM_CLIENTS = 100
CLIENTS_PER_ROUND = 10
LOCAL_EPOCHS = 5      
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(model, test_loader):
    model.eval()
    correct_all = 0
    correct_minor = 0
    total_minor = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct_all += pred.eq(target.view_as(pred)).sum().item()
            
            mask = target >= 5
            if mask.sum() > 0:
                minor_preds = pred[mask]
                minor_targets = target[mask]
                correct_minor += minor_preds.eq(minor_targets.view_as(minor_preds)).sum().item()
                total_minor += mask.sum().item()
                
    acc_all = correct_all / len(test_loader.dataset)
    acc_minor = correct_minor / total_minor if total_minor > 0 else 0
    return acc_all, acc_minor

def run_simulation(clients, test_loader, config):
    print(f"\n>>> Running Scenario: {config['name']}")
    
    # 3채널 고정
    global_model = SimpleCNN(num_channels=3).to(DEVICE)
    logs = []
    participation_count = np.zeros(NUM_CLIENTS)
    
    total_elapsed_time = 0.0
    acc_all, acc_minor = evaluate(global_model, DataLoader(test_loader, batch_size=1000))
    logs.append((0.0, acc_all, acc_minor, 0))

    target_time = config.get('target_time', 12000.0)
    server_momentum = 1.0 
    r = 0
    
    while total_elapsed_time < target_time:
        r += 1
        if r % 20 == 0: 
            print(f"\n[R{r}|T{total_elapsed_time:.0f}]", end="", flush=True)

        # 1. Selection
        if config['selection'] == 'sorted':
            all_times = [(c, c.estimate_time_with_epochs(LOCAL_EPOCHS)) for c in clients]
            all_times.sort(key=lambda x: x[1])
            candidates = all_times[:CLIENTS_PER_ROUND]
        else:
            candidates_clients = random.sample(clients, CLIENTS_PER_ROUND)
            candidates = [(c, c.estimate_time_with_epochs(LOCAL_EPOCHS)) for c in candidates_clients]
        
        current_deadline = config['fixed_deadline']
        selected_clients = []

        # 2. Rescue
        for client, t_full in candidates:
            if t_full <= current_deadline:
                selected_clients.append((client, LOCAL_EPOCHS))
            else:
                if config['rescue_mode']:
                    selected_clients.append((client, 3)) 
                else:
                    pass 

        if not selected_clients:
            total_elapsed_time += current_deadline
            logs.append((total_elapsed_time, acc_all, acc_minor, 0))
            continue

        # 3. Training
        local_weights = []
        actual_times = []
        effective_client_count = 0 
        global_w = global_model.state_dict()

        for client, epochs in selected_clients:
            w, real_ep = client.train(global_w, epochs)
            est_t = client.estimate_time_with_epochs(real_ep)
            act_t = est_t * random.uniform(0.95, 1.05)
            
            if act_t <= current_deadline:
                local_weights.append(w)
                actual_times.append(act_t)
                participation_count[client.id] += 1
                effective_client_count += 1
        
        if not local_weights:
            total_elapsed_time += current_deadline
            logs.append((total_elapsed_time, acc_all, acc_minor, 0))
            continue

        # 4. Aggregation
        avg_w = copy.deepcopy(global_w)
        num_participants = len(local_weights)
        for k in avg_w.keys():
            temp_sum = torch.stack([lw[k] for lw in local_weights]).sum(dim=0)
            avg_w[k] = temp_sum / num_participants
        
        final_w = copy.deepcopy(global_w)
        for k in final_w.keys():
            final_w[k] = (1 - server_momentum) * global_w[k] + server_momentum * avg_w[k]
        global_model.load_state_dict(final_w)

        round_time = max(actual_times) if actual_times else current_deadline
        total_elapsed_time += round_time
        
        acc_all, acc_minor = evaluate(global_model, DataLoader(test_loader, batch_size=1000))
        logs.append((total_elapsed_time, acc_all, acc_minor, effective_client_count))
        
        if r % 20 == 0: 
            print(f" Acc:{acc_all:.4f}", end="", flush=True)

    return logs, participation_count, global_model

if __name__ == "__main__":
    print(f">>> [System] Device: {DEVICE}, Dataset: {DATASET_NAME}")
    set_seed(42)
    
    # Init
    client_datasets, test_data = get_dataset_realistic_noniid(DATASET_NAME, NUM_CLIENTS)
    test_loader = DataLoader(test_data, batch_size=1000)

    clients = []
    for i in range(NUM_CLIENTS):
        c = Client(i, client_datasets[i], DEVICE, DATASET_NAME)
        clients.append(c)
    
    save_dir = os.path.join('results', DATASET_NAME)
    os.makedirs(save_dir, exist_ok=True)

    # Scenarios
    scenarios = [
        {'name': 'FedCS', 'selection': 'sorted', 'fixed_deadline': 100.0, 'rescue_mode': False},
        {'name': 'FedLim', 'selection': 'random', 'fixed_deadline': 100.0, 'rescue_mode': False},
        {'name': 'Proposed', 'selection': 'random', 'fixed_deadline': 100.0, 'rescue_mode': True}
    ]

    results = {}
    client_stats = {}
    per_class_accs = {}

    for config in scenarios:
        config['target_time'] = 12000.0
        set_seed(42)
        log, stats, model = run_simulation(clients, test_data, config)
        
        results[config['name']] = log
        client_stats[config['name']] = stats
        
        print(f"   >>> Calculating Per-Class Acc for {config['name']}...")
        model.eval()
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                outputs = model(data)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == target).squeeze()
                if c.ndim == 0:
                    label = target.item()
                    class_correct[label] += c.item()
                    class_total[label] += 1
                else:
                    for j in range(len(target)):
                        label = target[j]
                        class_correct[label] += c[j].item()
                        class_total[label] += 1
        
        acc_list = []
        for c, t in zip(class_correct, class_total):
            acc_list.append(c / t if t > 0 else 0.0)
        per_class_accs[config['name']] = acc_list

    save_path = os.path.join(save_dir, "experiment_data.pkl")
    with open(save_path, "wb") as f:
        pickle.dump({
            "results": results,
            "client_stats": client_stats,
            "per_class_accs": per_class_accs,
            "scenarios": scenarios
        }, f)
        
    print(f"\n>>> Experiment Finished. Data saved to {save_path}")