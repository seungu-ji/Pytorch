import os
import numpy as np
import matplotlib.pyplot as plt

result_dir = './results/numpy'

lst_data = os.listdir(result_dir)

lst_label = [f for f in lst_data if startswith('label')]
lst_input = [f for f in lst_data if startswith('input')]
lst_output = [f for f in lst_data if startswith('output')]

lst_label.sort()
lst_input.sort()
lst_output.sort()

## 0th data load
id = 0

label = np.load(os.path.join(result_dir, lst_label[id]))
input = np.load(os.path.join(result_dir, lst_input[id]))
output = np.load(os.path.join(result_dir, lst_output[id]))

## visualization
plt.subplot(131)
plt.imshow(label, cmap='gray')
plt.title('label')

plt.subplot(132)
plt.imshow(input, cmap='gray')
plt.title('input')

plt.subplot(133)
plt.imshow(output, cmap='gray')
plt.title('output')