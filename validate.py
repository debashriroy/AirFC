import numpy as np
from scipy.io import loadmat,savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from complexPyTorch.complexFunctions import complex_relu
# import os
# import ipdb; ipdb.set_trace()
# input_path = 'matfiles/to_send_to_Kubra2/'
input_path = 'matfiles/Noise_without_ReLU_matches/'
total_examples = 153

# read the data
# model_inputs = loadmat(input_path+'model_weights_16.mat')
model_inputs = loadmat(input_path+'model_inputs_16_noise.mat')
logical_out = model_inputs['out_x'][0:total_examples]
original_out = model_inputs['y'][0:total_examples]  # original data out
air_out = np.zeros((total_examples,64,10), dtype=np.complex64) # OTA out

for i in range(0,total_examples):
    ex = loadmat(input_path+'OTA_data\\example'+str(i)+'.mat')
    air_out[i] = ex['out_symbols']

print(air_out.shape)
print(logical_out.shape)

# VALIDATE NET
class ValidateNet(nn.Module):

    def __init__(self):
        super(ValidateNet, self).__init__()

    def forward(self, x):
        # x = complex_relu(x)
        x = x.abs()
        x = F.log_softmax(x, dim=1)
        # print('Final Shape: ', x.shape)
        return x


# SETTING UP MODEL PARAMETERS
os.environ['PYTHONHASHSEED']=str(1234)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("CUDA AVAILABILITY: ", torch.cuda.is_available())
model = ValidateNet().to(device)

def test(model, device):
    model.eval()

    running_loss = 0
    correct_logic = 0
    correct_original = 0
    total = 0

    with torch.no_grad():
        # for batch_idx, (data, target) in enumerate(test_loader):
        for i in range (total_examples):

            # data, target = data.to(device).type(torch.complex64), target.to(device)
            target1 = model(torch.from_numpy(logical_out[i]))
            target2 = torch.from_numpy(original_out[i]) # the original output
            outputs= model(torch.from_numpy(air_out[i]))

            target1 = torch.max(target1, 1)[1]
            predicted = torch.max(outputs, 1)[1]
            total += target1.size(0)

            correct_logic += predicted.eq(target1).sum().item()
            correct_original += predicted.eq(target2).sum().item()
            print(i, target1.size(0), predicted.eq(target2).sum().item())

    # test_loss = running_loss / total_examples
    accu_logic = 100. * correct_logic / total
    accu_original = 100. * correct_original / total

    # eval_losses.append(test_loss)
    # eval_accu.append(accu)

    print('Accuracy of the OTA output compared with logically predicted output: %.3f' % (accu_logic))
    print('Accuracy of the OTA output compared with original output: %.3f' % (accu_original))

# calculate the accuracy
test(model, device)

