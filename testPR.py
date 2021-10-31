
import torch
import torchvision
from models.backbone.resnet_PR import *
from models.backbone.csp_darknet import CSPDarknet



from collections import OrderedDict
def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k,v in state_dict.items():
        name = ','.join(k.split('.')[start_idx:])
        new_state_dict[name] = v







# model = CSPDarknet() #自己定义的模型，但要保证前面保存的层和自定义的模型中的层一致    有2个key   epoch    state_dict


print('--------------------------------------------------------')


state_dict = torch.load('./pretrain/model_last.pth')# 模型
pre_state = state_dict["model"]

csp = CSPDarknet(out_indices=(5))
model_state = csp.state_dict()

# for k in csp.state_dict():
#     print(k)

for i in pre_state:
    print(i)



# model.load_state_dict(state_dict)











