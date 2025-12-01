import re

import matplotlib.pyplot as plt
from constants import LOG_FILE, LOG_PATTERN


def parse_log():
    iterations = []
    rewards = []
    kls = []
    entropies = []
    policies = []
    values = []

    regex = re.compile(LOG_PATTERN)

    with open(LOG_FILE, "r") as f:
        for line in f:
            match = regex.search(line)
            if match:
                _, it, reward, kl, ent, pol, val = match.groups()
                iterations.append(int(it))
                rewards.append(float(reward))
                kls.append(float(kl))
                entropies.append(float(ent))
                policies.append(float(pol))
                values.append(float(val))

    return iterations, rewards, kls, entropies, policies, values


def plot_curves():
    it, reward, kl, ent, pol, val = parse_log()

    plt.figure(figsize=(14, 8))

    plt.subplot(2, 3, 1)
    plt.plot(it, reward)
    plt.title("Reward")

    plt.subplot(2, 3, 2)
    plt.plot(it, kl)
    plt.title("KL Divergence")

    plt.subplot(2, 3, 3)
    plt.plot(it, ent)
    plt.title("Policy Entropy")

    plt.subplot(2, 3, 4)
    plt.plot(it, pol)
    plt.title("Policy Loss")

    plt.subplot(2, 3, 5)
    plt.plot(it, val)
    plt.title("Value Loss")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_curves()
