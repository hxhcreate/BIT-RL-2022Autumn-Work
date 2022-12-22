import torch
from q6_nature_torch import DeepQLearningNetWork

model_resnet101 = DeepQLearningNetWork(1,1,1,1)
model_resnet101.to("cpu")
# model_resnet101.load_state_dict({k.replace('module.',''):v for k,v in torch.load("weights/5score.weights")['state_dict'].items()})
order_dict= torch.load("weights/5score.weights", map_location='cpu')
print(order_dict['0.weight'])
for name, value in order_dict.items():
    print(name)
