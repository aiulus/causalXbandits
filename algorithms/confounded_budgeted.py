# @inproceedings{jamshidi2024confounded,
#   title={Confounded budgeted causal bandits},
#   author={Jamshidi, Fateme and Etesami, Jalal and Kiyavash, Negar},
#   booktitle={Causal Learning and Reasoning},
#   pages={423--461},
#   year={2024},
#   organization={PMLR}
# }

from math import sqrt, log
import numpy as np


def causal_ucb(env, budget, beta=1.0):
    t = 0
    arms = env.get_action_space()  # includes observational arm a0
    N = len([a for a in arms if a != "observe"])

    # Pull each arm once
    rewards = {a: [] for a in arms}
    counts = {a: 0 for a in arms}
    cost_spent = 0

    for a in arms:
        if cost_spent + env.get_cost(a) > budget:
            break
        y, _ = env.pull(a)
        rewards[a].append(y)
        counts[a] += 1
        cost_spent += env.get_cost(a)
        t += 1

    def upper_confidence_bound(mu_hat, n, t):
        return mu_hat + sqrt(2 * log(t + 1) / (n + 1))

    while cost_spent < budget:
        # Estimate means and UCBs
        ucbs = {}
        for a in arms:
            if counts[a] == 0:
                continue
            mu_hat = np.mean(rewards[a])
            ucb = upper_confidence_bound(mu_hat, counts[a], t)
            ucbs[a] = ucb / env.get_cost(a)

        # Pull the best action (by reward/cost UCB)
        best_arm = max(ucbs, key=ucbs.get)
        if cost_spent + env.get_cost(best_arm) > budget:
            break
        y, _ = env.pull(best_arm)
        rewards[best_arm].append(y)
        counts[best_arm] += 1
        cost_spent += env.get_cost(best_arm)
        t += 1

    return rewards, counts

def simple_regret_budgeted(env, budget):
    import numpy as np

    arms = env.get_action_space()
    a0 = "observe"
    B_half = budget // 2
    obs_data = []

    # Step 1: collect B/2 observational samples
    for _ in range(B_half):
        y, sample = env.pull(a0)
        obs_data.append((sample, y))

    # Step 2: estimate mu for all interventional arms using observational data
    mu_hats = {}
    freqs = {}
    for a in arms:
        if a == a0:
            mu_hats[a0] = np.mean([y for _, y in obs_data])
            continue
        i, x = env.parse_action(a)  # e.g., "do(X1=1)" -> ("X1", 1)
        match = [(s, y) for s, y in obs_data if s[i] == x]
        if match:
            mu_hats[a] = np.mean([y for _, y in match])
            freqs[a] = len(match) / B_half
        else:
            mu_hats[a] = 0.5  # uninformed
            freqs[a] = 0.0

    # Step 3: identify infrequent arms
    threshold = 1.0 / len(arms)
    infrequent_arms = [a for a in arms if a != a0 and freqs[a] < threshold]

    if not infrequent_arms:
        # fallback: continue observing
        for _ in range(B_half):
            y, _ = env.pull(a0)
            mu_hats[a0] = (mu_hats[a0] + y) / 2
        return max(mu_hats, key=mu_hats.get)

    # Step 4: explore infrequent arms equally
    arm_budget = B_half // len(infrequent_arms)
    for a in infrequent_arms:
        samples = []
        for _ in range(arm_budget):
            y, _ = env.pull(a)
            samples.append(y)
        mu_hats[a] = np.mean(samples)

    return max(mu_hats, key=mu_hats.get)
