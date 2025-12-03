import torch
import torch.distributions.transforms as transforms

dist = torch.distributions.Uniform(-1, 1)
dt = torch.distributions.TransformedDistribution(dist, [transforms.AffineTransform(0, 5)])

x = dt.sample((10,))
print(x)
log_prob = dt.log_prob(x)
print("log prob", log_prob)
print("prob", torch.exp(log_prob))

p = dist.log_prob(torch.tensor([-1, 1, 0]).unsqueeze(-1))
print("uniform log prob", p)
