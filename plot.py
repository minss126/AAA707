import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import argparse

# ==========================================
# 0. Configuration & Style
# ==========================================
PLOT_CONFIG = [
    {
        'key': 'FedLim',       
        'label': 'FedLim',     
        'color': '#A9A9A9',    # Dark Gray
        'linestyle': '-'       
    },
    {
        'key': 'FedCS',        
        'label': 'FedCS',
        'color': '#0F3ADA',    # Cobalt Blue
        'linestyle': '-'       
    },
    {
        'key': 'Proposed',
        'label': 'Proposed',
        'color': '#D21404',    # Classic Red
        'linestyle': '-'       
    }
]

def calculate_jains_index(participation_counts):
    counts = np.array(participation_counts)
    n = len(counts)
    sum_x = np.sum(counts)
    sum_sq_x = np.sum(np.square(counts))
    if sum_sq_x == 0: return 0.0
    return (sum_x ** 2) / (n * sum_sq_x)

# ==========================================
# 1. Plotting Functions
# ==========================================

# [Graph 1] Accuracy over Time
def plot_smoothed_results(results, save_dir, window_size=10):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    for config in PLOT_CONFIG:
        target_name = None
        for name in results.keys():
            if config['key'] in name:
                target_name = name
                break
        
        if target_name is None: continue

        log = results[target_name]
        try:
            unpacked = list(zip(*log))
            times = np.array(unpacked[0], dtype=float)
            accs_all = np.array(unpacked[1], dtype=float)
            accs_minor = np.array(unpacked[2], dtype=float)
        except: continue

        c = config['color']
        ls = config['linestyle']
        lbl = config['label']
        
        # Plot smoothed lines
        if len(times) >= window_size:
            w = np.ones(window_size)/window_size
            times_s = times[window_size-1:]
            acc_all_s = np.convolve(accs_all, w, 'valid')
            acc_min_s = np.convolve(accs_minor, w, 'valid')
            ax1.plot(times_s, acc_all_s, label=lbl, color=c, linestyle=ls, linewidth=3)
            ax2.plot(times_s, acc_min_s, label=lbl, color=c, linestyle=ls, linewidth=3)
        else:
            ax1.plot(times, accs_all, label=lbl, color=c, linestyle=ls, linewidth=3)
            ax2.plot(times, accs_minor, label=lbl, color=c, linestyle=ls, linewidth=3)

    def set_ax(ax, title):
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel("Time (s)", fontsize=14)
        ax.set_ylabel("Accuracy", fontsize=14)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize=12, loc='lower right' if 'Total' in title else 'upper left')
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}s'))
        ax.set_xlim(left=0)

    set_ax(ax1, "Total Accuracy")
    set_ax(ax2, "Minor Class Accuracy (Label 5-9)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Graph_Accuracy.png"), dpi=300)

# [Graph 2] Client Participation Histogram
def plot_participation(client_stats, save_dir):
    plt.figure(figsize=(12, 6))
    
    ordered_stats = []
    for config in PLOT_CONFIG:
        for name, stats in client_stats.items():
            if config['key'] in name:
                ordered_stats.append((config, stats))
                break
    
    bar_width = 0.8 / len(ordered_stats)
    indices = np.arange(100) 

    for i, (config, stats) in enumerate(ordered_stats):
        plt.bar(indices + (i * bar_width), stats, width=bar_width, 
                label=config['label'], color=config['color'], alpha=1.0)

    plt.axvline(x=60, color='gray', linestyle='--', label='Fast/Slow Boundary')
    plt.title("Client Participation Distribution", fontsize=16)
    plt.xlabel("Client ID", fontsize=14)
    plt.ylabel("Participation Count", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Graph_Participation.png"), dpi=300)

# [Graph 3] Effective Client Count (Restore!)
def plot_client_count(results, save_dir):
    plt.figure(figsize=(10, 6))
    
    for config in PLOT_CONFIG:
        target_name = None
        for name in results.keys():
            if config['key'] in name:
                target_name = name
                break
        if target_name is None: continue

        log = results[target_name]
        try:
            unpacked = list(zip(*log))
            times = np.array(unpacked[0], dtype=float)
            counts = np.array(unpacked[3], dtype=float) # Index 3 is client count
        except: continue

        ls = config['linestyle']
            
        # Smoothing for better visualization
        window = 10
        if len(counts) >= window:
            counts_s = np.convolve(counts, np.ones(window)/window, mode='valid')
            times_s = times[window-1:]
            plt.plot(times_s, counts_s, label=config['label'], color=config['color'], 
                     linewidth=2, linestyle=ls, alpha=0.9) 
        else:
            plt.plot(times, counts, label=config['label'], color=config['color'], 
                     linewidth=2, linestyle=ls, alpha=0.9)

    plt.title("Effective Number of Clients per Round", fontsize=16)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Number of Clients", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}s'))
    plt.legend(loc='center right') 
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Graph_Client_Count.png"), dpi=300)

# [Graph 4] Per-Class Accuracy
def plot_per_class_accuracy(per_class_accs, save_dir):
    plt.figure(figsize=(12, 6))
    classes = np.arange(10)
    
    ordered_accs = []
    for config in PLOT_CONFIG:
        for name, accs in per_class_accs.items():
            if config['key'] in name:
                ordered_accs.append((config, accs))
                break

    bar_width = 0.8 / len(ordered_accs)
    
    for i, (config, accs) in enumerate(ordered_accs):
        plt.bar(classes + (i * bar_width), accs, width=bar_width, 
                label=config['label'], color=config['color'], alpha=1.0)

    plt.xlabel('Class ID', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title('Per-Class Accuracy (Final Model)', fontsize=16)
    plt.xticks(classes + bar_width, classes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Graph_Per_Class_Accuracy.png"), dpi=300)

# ==========================================
# Main Logic
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    args = parser.parse_args()
    
    load_path = os.path.join("results", args.dataset, "experiment_data.pkl")
    
    if not os.path.exists(load_path):
        print(f"Error: Data file not found at {load_path}")
        print("Run main.py first.")
        exit()
        
    save_dir = os.path.dirname(load_path)
    print(f">>> Loading data from {load_path}...")

    with open(load_path, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    client_stats = data["client_stats"]
    per_class_accs = data.get("per_class_accs", {})

    # Generate Graphs
    print(">>> Generating Graphs...")
    plot_smoothed_results(results, save_dir)
    plot_participation(client_stats, save_dir)
    plot_client_count(results, save_dir)  # <--- [복구됨]
    
    if per_class_accs:
        plot_per_class_accuracy(per_class_accs, save_dir)
    
    print(f">>> Graphs saved in: {save_dir}")