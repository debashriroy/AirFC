import numpy as np
from scipy.io import loadmat,savemat
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexConv1d
from complexPyTorch.complexLayers_yanyu import  ComplexLinear, ComplexLinearNoise
import argparse

parser = argparse.ArgumentParser(description='Configure the parameters for validating the OTA output for AirCNN project.')
parser.add_argument('--setting', type=str, default='e2e', choices = ['fc', 'out', 'e2e'],
help='Which layer is being validated through the over-the-air transmission: fc (first layer), out (second layer), e2e (both layers).')

args = parser.parse_args()
print('Argument parser inputs', args)


from complexPyTorch.complexFunctions import complex_relu
# import os
# import ipdb; ipdb.set_trace()
# input_path = 'matfiles/to_send_to_Kubra2/'
input_path = 'matfiles/Noise_without_ReLU_matches/'
total_examples = 153
input_file_name = 'model_inputs_16_noise'
weight_file_name = 'model_weights_16_noise'

# read the data
# model_inputs = loadmat(input_path+'model_weights_16.mat')
model_inputs = loadmat(input_path+input_file_name+'.mat')
model_weights = loadmat(input_path+weight_file_name+'.mat')
for key in model_weights.keys():
    print(key)

if args.setting == 'fc': logical_out = model_inputs['hidden_x'][0:total_examples] # OTA out  for re-transmission
else: logical_out = model_inputs['out_x'][0:total_examples]
original_out = model_inputs['y'][0:total_examples]  # original data out
if args.setting == 'fc': air_out = np.zeros((total_examples,64,16), dtype=np.complex64) # OTA out  for re-transmission
else: air_out = np.zeros((total_examples,64,10), dtype=np.complex64) # OTA out

for i in range(0,total_examples):
    if args.setting == 'out': ex = loadmat(input_path+'OTA_data\\example'+str(i)+'.mat')
    elif args.setting == 'fc': ex = loadmat(input_path + 'FC_input_16Tx_re_noise2\\example' + str(i) + '.mat')
    else: ex = loadmat(input_path + 'FC_final_16Tx_noise2\\example' + str(i) + '.mat')
    air_out[i] = ex['out_symbols']

print(air_out.shape)
print(logical_out.shape)

# VALIDATE NET
class ValidateNetOUT(nn.Module):

    def __init__(self):
        super(ValidateNetOUT, self).__init__()

    def forward(self, x):
        # x = complex_relu(x)

        x = x.abs()
        x = F.log_softmax(x, dim=1)
        # print('Final Shape: ', x.shape)
        return x

# VALIDATE NET
class ValidateNetFC(nn.Module):

    def __init__(self):
        super(ValidateNetFC, self).__init__()
        # self.out = ComplexLinear(16, 10) # ADDED TO TEST RE-TRANSMISSION ACCURACY

        self.out = ComplexLinearNoise(16, 10, realNoiseVar=1e-4, imgNoiseVar=1e-4, realWeightVar=1e-4,
                                      imgWeightVar=1e-4,
                                      quant=False,
                                      pruning=False,
                                      pruningRate=0.5,
                                      setSize=16, bias=False)  # ADDED TO TEST RE-TRANSMISSION ACCURACY

    def forward(self, x):
        # x = complex_relu(x)

        # x = self.out(x)  # ADDED TO TEST RE-TRANSMISSION ACCURACY
        # x = x.abs()
        xr, xi = self.out(torch.real(x), torch.imag(x)) # ADDED TO TEST RE-TRANSMISSION ACCURACY
        x = torch.sqrt(torch.pow(xr, 2) + torch.pow(xi, 2)) # ADDED TO TEST RE-TRANSMISSION ACCURACY

        x = F.log_softmax(x, dim=1)
        # print('Final Shape: ', x.shape)
        return x


# SETTING UP MODEL PARAMETERS
# os.environ['PYTHONHASHSEED']=str(1234)
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     # The GPU id to use
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
# print("CUDA AVAILABILITY: ", torch.cuda.is_available())
if args.setting == 'fc': model = ValidateNetFC().to(device)
else: model = ValidateNetOUT().to(device)
# else:  model = ValidateNetOUT().to(device)
pred_y = []
print(model)
# print(model.out)

def test(model, device):
    model.eval()
    running_loss = 0
    correct_logic = 0
    correct_original = 0
    total = 0

    with torch.no_grad():
        if args.setting == 'fc':
            model.out.fc_r.weight = nn.Parameter(torch.from_numpy(model_weights['out.fc_r.weight']))
            model.out.fc_i.weight = nn.Parameter(torch.from_numpy(model_weights['out.fc_i.weight']))
        # for batch_idx, (data, target) in enumerate(test_loader):
        for i in range (total_examples):

            # data, target = data.to(device).type(torch.complex64), target.to(device)
            target1 = model(torch.from_numpy(logical_out[i]))
            target2 = torch.from_numpy(original_out[i]) # the original output
            outputs= model(torch.from_numpy(air_out[i]))

            target1 = torch.max(target1, 1)[1]
            predicted = torch.max(outputs, 1)[1]
            pred_y.append(predicted.cpu().detach().numpy())
            total += target1.size(0)

            correct_logic += predicted.eq(target1).sum().item()
            correct_original += predicted.eq(target2).sum().item()
            # print(i, target1.size(0), predicted.eq(target2).sum().item())

    # test_loss = running_loss / total_examples
    accu_logic = 100. * correct_logic / total
    accu_original = 100. * correct_original / total

    # eval_losses.append(test_loss)
    # eval_accu.append(accu)

    print('Accuracy of the OTA output compared with logically predicted output: %.3f' % (accu_logic))
    print('Accuracy of the OTA output compared with original output: %.3f' % (accu_original))

# calculate the accuracy
test(model, device)

###########################
# save the inputs
##########################
print("Model's state_dict:")
input_dic = {}
# input_dic['input_x'] = input_x.cpu().detach().numpy()
# input_dic['hidden_x'] = hidden_x.cpu().detach().numpy()
# input_dic['out_x'] = out_x.cpu().detach().numpy()
input_dic['input_x'] = model_inputs['input_x'][0:total_examples]
input_dic['hidden_x'] = model_inputs['hidden_x'][0:total_examples]
input_dic['out_x'] = model_inputs['out_x'][0:total_examples]
input_dic['y'] = model_inputs['y'][0:total_examples]
input_dic['ota_y'] = pred_y


if args.setting == 'out':
    stored_file = input_path + input_file_name + "_out_with_prediction.mat"
elif args.setting == 'fc':
    stored_file = input_path + input_file_name + "_fc_noise2_with_prediction.mat"
else:
    stored_file = input_path + input_file_name + "_e2e_noise2_with_prediction.mat"
savemat(stored_file, input_dic)
from scipy.io import loadmat
model_inputs_evaluate = loadmat(stored_file)
# for key in model_inputs.keys():
#     print(key, np.array(model_inputs[key]))
for key in model_inputs_evaluate.keys():
    print(key, np.array(model_inputs_evaluate[key]).shape)
print("Input Keys are: ", model_inputs_evaluate.keys())

# TESTING IF THE SAVED PREDICTION IS CORRECT
saved_air_out = model_inputs_evaluate['ota_y'][0:total_examples]
total = 0
correct = 0
for i in range(total_examples):
    target2 = torch.from_numpy(original_out[i])  # the original output
    predicted = torch.from_numpy(saved_air_out[i])

    # predicted = torch.max(outputs, 1)[1]
    total += target2.size(0)

    correct += predicted.eq(target2).sum().item()
    # print(i, target1.size(0), predicted.eq(target2).sum().item())

# test_loss = running_loss / total_examples
# print(100. * correct / total)
print('Accuracy of the OTA output compared with original output (SAVED IN THE MATFILE): %.3f' % (100. * correct / total))

