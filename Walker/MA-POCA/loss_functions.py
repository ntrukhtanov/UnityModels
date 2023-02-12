import torch


def calc_value_loss(values, old_values, returns, epsilon):
    clipped_value_estimate = old_values + torch.clamp(
        values - old_values, -1 * epsilon, epsilon
    )
    v_opt_a = (returns - values) ** 2
    v_opt_b = (returns - clipped_value_estimate) ** 2
    loss = torch.mean(torch.max(v_opt_a, v_opt_b))
    return loss
