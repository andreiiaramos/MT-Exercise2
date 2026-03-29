#!/usr/bin/env python3
import sys
import argparse
import csv
import matplotlib.pyplot as plt

def parse_test_ppl(log_paths):
    import re
    results = {}
    for path in log_paths:
        label = path.split('/')[-1].replace('.log', '')
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                m = re.search(r'test ppl\s+([\d.]+)', line, re.IGNORECASE)
                if m:
                    results[label] = float(m.group(1))
                    break
    return results

def print_test_table(test_data):
    print("\nTest Perplexity Table")
    print("Model,Test PPL")
    for label, ppl in sorted(test_data.items()):
        print(f"{label},{ppl:.2f}")
    print()
def process_logs(log_paths):
    data = {} 
    for path in log_paths:
        # Use filename as the label (e.g., 'log_dp0.3')
        label = path.split('/')[-1].replace('.csv', '').replace('.log', '')
        data[label] = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                ep = int(row['epoch'])
                data[label][ep] = {
                    'train': float(row['train_ppl']),
                    'valid': float(row['valid_ppl'])
                }
    return data

def print_table(data, metric, title):
    print(f"\n{title}")
    labels = list(data.keys())
    epochs = sorted(list(data[labels[0]].keys())) if labels else []
    
    header = ["Epoch"] + labels
    print(",".join(header))
    
    for ep in epochs:
        row = [str(ep)]
        for label in labels:
            val = data[label].get(ep, {}).get(metric, "N/A")
            if val != "N/A":
                row.append(f"{val:.2f}")
            else:
                row.append(str(val))
        print(",".join(row))
    print("\n")

def plot_metric(data, metric, out_path, title):
    plt.figure(figsize=(8,5))
    for label, eps in data.items():
        x = sorted(list(eps.keys()))
        y = [eps[ep][metric] for ep in x]
        plt.plot(x, y, label=label)
        
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"Wrote plot: {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('logs', nargs='+', help='CSV log files')
    p.add_argument('--stdout-logs', nargs='+', help='stdout run logs for test PPL extraction')
    p.add_argument('--out-train', default='train_plot.png')
    p.add_argument('--out-valid', default='valid_plot.png')
    args = p.parse_args()

    data = process_logs(args.logs)

    print_table(data, 'train', 'Training Perplexity Table')
    print_table(data, 'valid', 'Validation Perplexity Table')

    if args.stdout_logs:
        test_data = parse_test_ppl(args.stdout_logs)
        if test_data:
            print_test_table(test_data)

    plot_metric(data, 'train', args.out_train, 'Training Perplexity over Epochs')
    plot_metric(data, 'valid', args.out_valid, 'Validation Perplexity over Epochs')

if __name__=='__main__':
    main()



