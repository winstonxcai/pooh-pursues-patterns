#!/usr/bin/env python3
"""
Plot training metrics from DPO training log file.
"""

import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from constants import LOG_PATTERN


def parse_log_file(log_path):
    """Extract (step, loss, kl_div) tuples from log file."""
    steps = []
    losses = []
    kl_divs = []
    
    # Use pattern from constants file
    pattern = LOG_PATTERN
    
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                kl_div = float(match.group(3))
                steps.append(step)
                losses.append(loss)
                kl_divs.append(kl_div)
    
    return steps, losses, kl_divs


def plot_metrics(steps, losses, kl_divs, output_dir=None):
    """Plot loss and KL divergence over time."""
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Loss (log scale)
    log_losses = np.log(np.array(losses) + 1e-8)
    axes[0].plot(steps, log_losses, linewidth=1.5, alpha=0.7, color='blue')
    axes[0].set_xlabel('Step', fontsize=12)
    axes[0].set_ylabel('Log Loss', fontsize=12)
    axes[0].set_title('Training Log Loss Over Time', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: KL Divergence
    axes[1].plot(steps, kl_divs, linewidth=1.5, alpha=0.7, color='green')
    axes[1].set_xlabel('Step', fontsize=12)
    axes[1].set_ylabel('KL Divergence', fontsize=12)
    axes[1].set_title('KL Divergence Over Time', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        output_path = f"{output_dir}/dpo_training_metrics.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    if len(sys.argv) < 2:
        log_file = "dpo_training.log"
        print(f"No log file specified, using default: {log_file}")
    else:
        log_file = sys.argv[1]
    
    try:
        steps, losses, kl_divs = parse_log_file(log_file)
        
        if not steps:
            print(f"No training data found in {log_file}")
            return
        
        print(f"Parsed {len(steps)} data points")
        print(f"Step range: {min(steps)} - {max(steps)}")
        print(f"Loss range: {min(losses):.4f} - {max(losses):.4f}")
        print(f"KL Div range: {min(kl_divs):.4f} - {max(kl_divs):.4f}")
        
        # Use log file directory as output directory
        import os
        output_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else "."
        plot_metrics(steps, losses, kl_divs, output_dir)
        
    except FileNotFoundError:
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

