import torch
from torch import inf
import torch.distributions.utils

# only supports single counts unless total_count set


# print(torch.distributions.Binomial(probs=0.9).log_prob(torch.tensor([[0., 1.0], [0., 1.0]])))

# probs = torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
# counts = torch.tensor([[0., 0.0, 2.0], [0.0, 1.0, 1.0]])
# print(torch.distributions.Multinomial(probs=probs, total_count=2).log_prob(counts))


# probs = torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
# counts = torch.tensor([[0., 0.0, 3.0], [0.0, 2.0, 1.0]])  # 3
# print(torch.distributions.Multinomial(probs=probs, total_count=3).log_prob(counts))

# def get_log_prob_of_single_galaxy(probs, counts):
#     return torch.distributions.Multinomial(probs=probs, total_count=2).log_prob(counts)
    
# get_log_prob_of_galaxies = torch.vmap(get_log_prob_of_single_galaxy, in_dims=0, out_dims=0)

def get_multinomial_log_prob(probs, counts):
        # _categorical = torch.distributions.Categorical(probs=probs)
        # logits = _categorical.logits
        logits = torch.distributions.utils.probs_to_logits(probs)
        # logits, counts = broadcast_all(self.logits, value)
        log_factorial_n = torch.lgamma(counts.sum(-1) + 1)
        log_factorial_xs = torch.lgamma(counts + 1).sum(-1)
        logits[(counts == 0) & (logits == -inf)] = 0
        log_powers = (logits * counts).sum(-1)
        return log_factorial_n - log_factorial_xs + log_powers

# probs = torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
# counts = torch.tensor([[0., 0.0, 1.0], [0.0, 0.0, 1.0]])
# # print(probs.shape, counts.shape)
# print(get_log_prob_of_galaxies(probs, counts))

# print(get_multinomial_log_prob(probs, counts))

# probs = torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
# counts = torch.tensor([[0.0, 0.0, 3.0], [0.0, 1.0, 1.0]])
# print(get_multinomial_log_prob(probs, counts))

# probs = torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
# counts = torch.tensor([[0.0, 0.0, 3.0], [0.0, 1.0, 2.0]])

probs = torch.tensor([[0.4, 0.2, 0.4], [0.2, 0.4, 0.4]])
counts = torch.tensor([[0.0, 1.0, 3.0], [0.0, 1.0, 7.0]])

for p, c in zip(probs, counts):
    # print(c.sum())
    total_count = int(c.sum().numpy().squeeze())
    print(torch.distributions.Multinomial(probs=p, total_count=total_count).log_prob(c))
    print(get_multinomial_log_prob(p, c))

print(get_multinomial_log_prob(probs, counts))