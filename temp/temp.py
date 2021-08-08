import torch
import torch.nn.functional as F

boxes = torch.tensor([0.1, 0.3, 0.5, 1.0])
mask = torch.ByteTensor([[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]])
b = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4]])
# x = torch.tensor([4, 4, 4, 4])
# b = b / x
print(boxes)
print(mask)
print(b)

x0, y0 = boxes[0], boxes[1]
ww, hh = boxes[2], boxes[3]
# print(x0)
# print(y0)
# print(ww)
# print(hh)

X = torch.linspace(0, 1, steps=7).view(1, 1, 7).to(boxes)
Y = torch.linspace(0, 1, steps=7).view(1, 7, 1).to(boxes)
# print(X)
# print(Y)

X = (X - x0) / ww  # (1, W)
Y = (Y - y0) / hh  # (H, 1)
# print(X)
# print(Y)

X = X.expand(1, 7, 7)
Y = Y.expand(1, 7, 7)
grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)
# print(X)
# print(Y)
# print(grid)

# Right now grid is in [0, 1] space; transform to [-1, 1]
grid = grid.mul(2).sub(1)

img_in = b.float().view(1, 1, 4, 4)
sampled = F.grid_sample(img_in, grid, mode='bilinear')
print(sampled)

out = sampled * mask
print(out)
