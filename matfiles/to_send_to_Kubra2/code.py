import numpy as np
from scipy.io import loadmat
model_weights = loadmat('model_weights_16.mat') # Give the corresponding name
# for key in model_weights.keys():
#     # print(key, model_weights[key])
#     print("type: ", type(model_weights[key]))
print("Model Keys are: ", model_weights.keys())
weight_keys = list(model_weights.keys())
print("Shape of the model weights: ", model_weights[weight_keys[5]].shape, model_weights[weight_keys[6]].shape)

model_inputs = loadmat('model_inputs_16.mat') # Give the corresponding name
# for key in model_inputs.keys():
#     # print(key, model_inputs[key])
#     print("type: ", type(model_inputs[key]))
print("Input Keys are: ", model_inputs.keys())
input_keys = list(model_inputs.keys())
print("Shape of the model inputs: ", model_inputs[input_keys[3]].shape, model_inputs[input_keys[4]].shape, model_inputs[input_keys[5]].shape)

last_example = model_inputs[input_keys[3]].shape[0] -1 # Taking example from the last trained epoch
input = model_inputs[input_keys[3]][last_example]
hidden = model_inputs[input_keys[4]][last_example]
output = model_inputs[input_keys[5]][last_example]

print("Model input: ", input)
print("Hidden layer output: ", hidden)
print("Output : ", output)

########### Calculation on hidden to output layer ###################
hidden_to_out_weight_matrix_real = model_weights[weight_keys[5]]
hidden_to_out_weight_matrix_imag = model_weights[weight_keys[6]]

print("Model Weights of Hidden to Output Real Part: ", hidden_to_out_weight_matrix_real)
print("Model Weights of Hidden to Output Imag Part: ", hidden_to_out_weight_matrix_imag)


hidden_to_out_weight_matrix = hidden_to_out_weight_matrix_real + 1j * hidden_to_out_weight_matrix_imag # converting the real and imag weight matrix to a complex one
multiplied_out = np.matmul(hidden_to_out_weight_matrix, np.transpose(hidden)) #  y = W * X'
print('********* The multiplied output: ', multiplied_out)
print('********* The actual output: ',np.transpose(output))
print("The Shapes of multiplied output and actual output: ", multiplied_out.shape, np.transpose(output).shape)
########### End of calculation on hidden to output layer ###################


########### Calculation on input to hidden layer ###################
input_to_hidden_weight_matrix_real = model_weights[weight_keys[3]]
input_to_hidden_weight_matrix_imag = model_weights[weight_keys[4]]

print("Model Weights of Input to Hidden Real Part: ", input_to_hidden_weight_matrix_real)
print("Model Weights of Input to Hidden Imag Part: ", input_to_hidden_weight_matrix_imag)


input_to_hidden_weight_matrix = input_to_hidden_weight_matrix_real + 1j * input_to_hidden_weight_matrix_imag # converting the real and imag weight matrix to a complex one
multiplied_hidden = np.matmul(input_to_hidden_weight_matrix, np.transpose(input)) #  y = W * X'
print('********* The multiplied hidden: ', multiplied_hidden)
print('********* The actual hidden: ', np.transpose(hidden))
print("The Shapes of multiplied hidden and actual hidden: ", multiplied_hidden.shape, np.transpose(hidden).shape)
########### End of calculation on input to hidden layer ###################