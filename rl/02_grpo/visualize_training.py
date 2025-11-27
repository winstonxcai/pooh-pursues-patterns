#!/usr/bin/env python3
"""
Plot training loss from GRPO training log file.
"""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np


def parse_log_file(log_path):
    """Extract (step, loss) pairs from log file."""
    steps = []
    losses = []
    
    # Pattern to match: "step 4590 | avg_loss (last 10 steps): 7.8106"
    pattern = r'step (\d+) \| avg_loss \(last \d+ steps\): ([\d.]+)'
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                steps.append(step)
                losses.append(loss)
    
    return steps, losses

def plot_loss(steps, losses, output_path=None):
    """Plot steps vs log loss."""
    log_losses = np.log(np.array(losses) + 1e-8)  # Add small epsilon to avoid log(0)
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, log_losses, linewidth=1.5, alpha=0.7)
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Log Average Loss', fontsize=12)
    plt.title('Training Log Loss Over Time', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

def main():
    if len(sys.argv) < 2:
        log_file = "grpo_training.log"
        print(f"No log file specified, using default: {log_file}")
    else:
        log_file = sys.argv[1]
    
    try:
        steps, losses = parse_log_file(log_file)
        
        if not steps:
            print(f"No loss data found in {log_file}")
            return
        
        print(f"Parsed {len(steps)} data points")
        print(f"Step range: {min(steps)} - {max(steps)}")
        print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
        print(f"Log loss range: {np.log(min(losses) + 1e-8):.4f} - {np.log(max(losses) + 1e-8):.4f}")
        
        output_path = log_file.replace('.log', '_log_loss_plot.png')
        plot_loss(steps, losses, output_path)
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

