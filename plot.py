import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as ticker
import argparse

# ==========================================
# 0. Configuration & Style
# ==========================================
# Professional High-Contrast Colors
PLOT_CONFIG = [
    {
        'key': 'FedLim',       
        'label': 'FedLim',     
        'color': '#404040',    # Dark Gray
        'linestyle': '-'       
    },
    {
        'key': 'FedCS',        
        'label': 'FedCS',
        'color': '#0047AB',    # Cobalt Blue
        'linestyle': '-'       
    },
    {
        'key': 'Proposed',
        'label': 'Proposed',
        'color': '#DA291C',    # Classic Red
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

        # Plot smoothed lines
        if len(times) >= window_size:
            w = np.ones(window_size)/window_size
            times_s = times[window_size-1:]
            acc_all_s = np.convolve(accs_all, w, 'valid')
            acc_min_s = np.convolve(accs_minor, w, 'valid')
            ax1.plot(times_s, acc_all_s, label=config['label'], color=config['color'], 
                     linestyle=config['linestyle'], linewidth=3)
            ax2.plot(times_s, acc_min_s, label=config['label'], color=config['color'], 
                     linestyle=config['linestyle'], linewidth=3)
        else:
            ax1.plot(times, accs_all, label=config['label'], color=config['color'], 
                     linestyle=config['linestyle'], linewidth=3)
            ax2.plot(times, accs_minor, label=config['label'], color=config['color'], 
                     linestyle=config['linestyle'], linewidth=3)

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
                label=config['label'], color=config['color'], alpha=1.0) # Solid Color

    plt.axvline(x=60, color='gray', linestyle='--', label='Fast/Slow Boundary')
    plt.title("Client Participation Distribution", fontsize=16)
    plt.xlabel("Client ID", fontsize=14)
    plt.ylabel("Participation Count", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Graph_Participation.png"), dpi=300)

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
                label=config['label'], color=config['color'], alpha=1.0) # Solid Color

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
        exit()
        
    save_dir = os.path.dirname(load_path)
    print(f">>> Loading data from {load_path}...")

    with open(load_path, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    client_stats = data["client_stats"]
    per_class_accs = data.get("per_class_accs", {})

    # Generate Summary Table
    print("\n" + "="*80)
    print(f"{'Method':<20} | {'Total Acc':<10} | {'Minor Acc':<10} | {'Fairness (Jain)':<15}")
    print("-" * 80)
    summary_text = ""
    for config in PLOT_CONFIG:
        target_name = None
        for name in client_stats.keys():
            if config['key'] in name:
                target_name = name
                break
        if target_name is None: continue

        stats = client_stats[target_name]
        fairness = calculate_jains_index(stats)
        last_log = results[target_name][-1]
        acc_total = last_log[1] * 100
        acc_minor = last_log[2] * 100
        
        line = f"{config['label']:<20} | {acc_total:.2f}%     | {acc_minor:.2f}%     | {fairness:.4f}"
        print(line)
        summary_text += line + "\n"
    print("="*80 + "\n")
    
    with open(os.path.join(save_dir, "Summary_Table.txt"), "w") as f:
        f.write(summary_text)

    # Generate Graphs
    print(">>> Generating Graphs...")
    plot_smoothed_results(results, save_dir)
    plot_participation(client_stats, save_dir)
    if per_class_accs:
        plot_per_class_accuracy(per_class_accs, save_dir)
    
    print(f">>> Graphs saved in: {save_dir}")