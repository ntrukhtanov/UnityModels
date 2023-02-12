import torch


def calc_value_loss(values, old_values, returns, epsilon):
    clipped_value_estimate = old_values + torch.clamp(
        values - old_values, -1 * epsilon, epsilon
    )
    v_opt_a = (returns - values) ** 2
    v_opt_b = (returns - clipped_value_estimate) ** 2
    loss = torch.mean(torch.max(v_opt_a, v_opt_b))
    return loss


def calc_returns(rewards, dones, values, gamma, lam, value_next):
    returns = torch.zeros_like(rewards)
    returns[-1] = rewards[-1] + gamma * value_next * (1 - dones[-1])
    for t in reversed(range(0, rewards.shape[0] - 1)):
        returns[t] = (
                gamma * lam * returns[t + 1] * (1 - dones[t + 1])
                + rewards[t]
                + (1 - lam) * gamma * values[t + 1] * (1 - dones[t + 1])
        )
    return returns
