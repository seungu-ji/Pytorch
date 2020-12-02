import numpy as np

import matplotlib.pyplot as plt 

## load image
img = plt.imread("Lenna.png")

# gray image generation
# img = np.mean(img, axis=2, keepdims=True)

size = img.shape

cmap = "gray" if size[2] == 1 else None

plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")
plt.show()

## Inpainting: Uniform sampling
ds_y = 2
ds_x = 4

mask = np.zeros(size)
mask[::ds_y, ::ds_x, :] = 1
# ds_x * ds_y matrix에 흰칸이 한개씩 있다.

dst = img * mask

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(mask), cmap=cmap, vmin=0, vmax=1)
plt.title("Uniform sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()

## Inpainting: Random sampling
rnd = np.random.rand(size[0], size[1], size[2])
# RGB 방향으로 동일한 sampling을 하고 싶을시
# rnd = np.random.rand(size[0], size[1], 1)

prob = 0.5

mask = (rnd > prob).astype(np.float)
# + mask = np.tile(mask, (1, 1, size[2]))

dst = img * mask

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(mask), cmap=cmap, vmin=0, vmax=1)
plt.title("Random sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()

## Inpainting: Gaussian sampling
ly = np.linspace(-1, 1, size[0])
lx = np.linspace(-1, 1, size[1])

x, y = np.meshgrid(lx, ly)

x0 = 0
y0 = 0
sgmx = 1
sgmy = 1

a = 1

gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
#plt.imshow(gaus)
gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, size[2]))

rnd = np.random.rand(size[0], size[1], size[2])

mask = (rnd < gaus).astype(np.float)

dst = img * mask

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(mask), cmap=cmap, vmin=0, vmax=1)
plt.title("Gaussian sampling mask")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Sampling image")

plt.show()