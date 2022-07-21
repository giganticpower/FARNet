import torch

checkpoint = "model/Ctw1500/textgraph_vgg_1.pth"

epoch = torch.load(checkpoint)
epoch1 = epoch['epoch']
print(epoch1)