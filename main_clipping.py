import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexConv1d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d
from scipy.io import savemat, loadmat
import json
import os
import argparse

import sys

parser = argparse.ArgumentParser(description='Configure the files before training the net.')
parser.add_argument('--id_gpu', default=1, type=int, help='which gpu to use.')
parser.add_argument('--epochs', default=100, type = int, help='Specify the epochs to train')
parser.add_argument('--lr', default=0.0001, type=float,help='learning rate for Adam optimizer',)
parser.add_argument('--bs',default=64, type=int,help='Batch size')
parser.add_argument('--hidden_elements', help='Number of hidden elements in the complex neural network architecture', type=int,default = 16)
# parser.add_argument('--model_file_name', help='Name of trained model file', type=str,default = 'model_weights')
# parser.add_argument('--data_file_name', help='Name of the file storing the inputs', type=str,default = '')

args = parser.parse_args()
print('Argumen parser inputs', args)


batch_size = args.bs
trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
train_set = datasets.MNIST('../data', train=True, transform=trans, download=True)
test_set = datasets.MNIST('../data', train=False, transform=trans, download=True)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last = True)





class WeightClipper(object):

    def __init__(self, frequency=5):
        self.frequency = frequency

    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, 'weight'):
            w = module.weight.data
            w = w.clamp(-1,1)


class ComplexNet(nn.Module):

    def __init__(self):
        super(ComplexNet, self).__init__()
        # self.conv1 = ComplexConv1d(4, 4, 1, 1) # in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias
        # self.conv2 = ComplexConv1d(4, 4, 1, 1)
        self.fc1 = ComplexLinear(784, args.hidden_elements, bias= False) # 784
        # self.fc2 = ComplexLinear(32, 32)
        self.out = ComplexLinear(args.hidden_elements, 10, bias= False)

    def forward(self, x):
        # x = torch.reshape(x, (x.shape[0], 4, 7*x.shape[3]))
        # print("Shape1: ", x.shape)
        # x = self.conv1(x)
        # print("Shape2: ", x.shape)
        # x = complex_relu(x)
        # x = complex_max_pool2d(x, 2, 2)
        # x = self.bn(x)
        # x = self.conv2(x)
        # x = complex_relu(x)
        # x = complex_max_pool2d(x, 2, 2)
        # print("Shape3: ", x.shape)
        inpt_x = x.view(x.shape[0], -1)
        # print("Shape after view: ", inpt_x.shape)
        hiddn_x = complex_relu(self.fc1(inpt_x)) # First hidden layer
        # print("Shape5: ", x.shape)
        # x = complex_relu(x)
        # x = self.fc2(x) # Second hidden layer
        # print("Values after first hidden layer: ", x)
        out_x = self.out(hiddn_x)
        # print("Shape after hidden layer: ", out_x.shape)
        # x = complex_relu(out_x)
        x = out_x.abs()
        x = F.log_softmax(x, dim=1)
        # print('Final Shape: ', x.shape)
        return x, inpt_x, hiddn_x, out_x

os.environ['PYTHONHASHSEED']=str(1234)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # The GPU id to use
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.id_gpu)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("CUDA AVAILABILITY: ", torch.cuda.is_available())
model = ComplexNet().to(device)
# model = ComplexNet()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
# optimizer = torch.optim.RMSprop(model.parameters(), lr=0.005, momentum=0.9) # same as nature paper
criterion = torch.nn.CrossEntropyLoss()
clipper = WeightClipper()
model.apply(clipper)


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).type(torch.complex64), target.to(device)
        optimizer.zero_grad()
        # print("Shape of Data: ", data.shape)
        output, _, _, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if epoch % clipper.frequency == 0:
            model.apply(clipper)
        if batch_idx % 100 == 0:
            print('Train Epoch: {:3} [{:6}/{:6} ({:3.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item())
            )


eval_losses = []
eval_accu = []

input_x_all = []
hidden_x_all = []
out_x_all = []
ground_truths = []

def test(model, device, test_loader, optimizer, epoch):
    model.eval()

    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device).type(torch.complex64), target.to(device)

            outputs, input_x, hidden_x, out_x = model(data)

            input_x_all.append(input_x.cpu().detach().numpy())
            hidden_x_all.append(hidden_x.cpu().detach().numpy())
            out_x_all.append(out_x.cpu().detach().numpy())

            # print("Individual shapes: ", input_x.cpu().detach().numpy().shape)
            # print("Individual shapes: ", hidden_x.cpu().detach().numpy().shape)
            # print("Individual shapes: ", out_x.cpu().detach().numpy().shape)

            loss = F.nll_loss(outputs, target)
            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            ground_truths.append(target.cpu().detach().numpy())

    test_loss = running_loss / len(test_loader)
    accu = 100. * correct / total

    eval_losses.append(test_loss)
    eval_accu.append(accu)

    print('Test Loss: %.3f | Accuracy: %.3f' % (test_loss, accu))

    # print("Total shapes: ", np.stack(input_x_all).shape)
    # print("total shapes: ", np.array(hidden_x_all).shape)
    # print("Total shapes: ", np.array(out_x_all).shape)

    return input_x_all, hidden_x_all, out_x_all, ground_truths



# Run training on 20 epochs
for epoch in range(args.epochs):
    train(model, device, train_loader, optimizer, epoch)

# testing after final round
input_x_all, hidden_x_all, out_x_all, ground_truths = test(model, device, test_loader, optimizer, epoch)

print("Shapes of returned elements: ", np.array(input_x_all).shape, np.array(input_x_all).shape, np.array(input_x_all).shape)

    # input_x_all.append(input_x.cpu().detach().numpy())
    # hidden_x_all.append(hidden_x.cpu().detach().numpy())
    # out_x_all.append(out_x.cpu().detach().numpy())

torch.save(model, 'mnist.pt')
print("Final MODEL: ")
print(model)

##############################
# Print model's state_dict
##############################
print("Model's state_dict:")
mdic = {}
for param_tensor in model.state_dict():
    # print(param_tensor, "\t", model.state_dict()[param_tensor].size(), "\t", model.state_dict()[param_tensor])
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    mdic[param_tensor] = model.state_dict()[param_tensor].cpu().detach().numpy()

savemat("matfiles/model_weights_"+str(args.hidden_elements)+".mat", mdic)
from scipy.io import loadmat
model_weights = loadmat("matfiles/model_weights_"+str(args.hidden_elements)+".mat")
# for key in model_weights.keys():
#     print(key, model_weights[key])
for key in model_weights.keys():
    print(key, np.array(model_weights[key]).shape)
print("Model Keys are: ", model_weights.keys())

###########################
# save the inputs
##########################
print("Model's state_dict:")
input_dic = {}
# input_dic['input_x'] = input_x.cpu().detach().numpy()
# input_dic['hidden_x'] = hidden_x.cpu().detach().numpy()
# input_dic['out_x'] = out_x.cpu().detach().numpy()
input_dic['input_x'] = input_x_all
input_dic['hidden_x'] = hidden_x_all
input_dic['out_x'] = out_x_all
input_dic['y'] = ground_truths


savemat("matfiles/model_inputs_"+str(args.hidden_elements)+".mat", input_dic)
from scipy.io import loadmat
model_inputs = loadmat("matfiles/model_inputs_"+str(args.hidden_elements)+".mat")
# for key in model_inputs.keys():
#     print(key, np.array(model_inputs[key]))
for key in model_inputs.keys():
    print(key, np.array(model_inputs[key]).shape)
print("Input Keys are: ", model_inputs.keys())
