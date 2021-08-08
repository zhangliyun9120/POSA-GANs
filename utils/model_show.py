import pickle
import torch


# pkl文件所在路径
path = '/tmp/OL-GANs/checkpoints/pretrained/models/iter-400000_netG_t.pkl'

f = open(path, 'rb')

model = torch.load(path)
# data = pickle.load(f)

print(model)

# print(len(data))
