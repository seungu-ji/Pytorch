import numpy as np
from scipy.stats import poisson
from scipy.io import loadmat

from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, rescale

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


## Inpainting

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
# RGB 방향으로 동일한 sampling을 하고 싶을시
# gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, 1))

rnd = np.random.rand(size[0], size[1], size[2])
# rnd = np.random(size[0], size[1], 1)

mask = (rnd < gaus).astype(np.float)
# + mask = np.tile(mask, (1, 1, size[2]))

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








## Denoising

## Denoising: Random noise
sgm = 60.0

noise = sgm/255.0 * np.random.randn(size[0], size[1], size[2])

dst = img + noise

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title("Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Noisy image")

plt.show()

## Denoising: Poisson noise (image-domain)
dst = poisson.rvs(255.0 * img) / 255.0
noise = dst - img

plt.subplot(131)
plt.imshow(np.squeeze(img), cmap=cmap, vmin=0, vmax=1)
plt.title("Ground Truth")

plt.subplot(132)
plt.imshow(np.squeeze(noise), cmap=cmap, vmin=0, vmax=1)
plt.title("Poisson Noise")

plt.subplot(133)
plt.imshow(np.squeeze(dst), cmap=cmap, vmin=0, vmax=1)
plt.title("Noisy image")

plt.show()

## Denoising: Poisson noise (CT-domain)
# system setting
N = 512
ANG = 180
VIEW = 360
THETA = np.linspace(0, ANG, VIEW, endpoint=False)

A = lambda x: radon(x, THETA, circle=False).astype(np.float32)
AT = lambda y: iradon(y, THETA, circle=False, filter=None, output_size=N).astype(np.float32)
AINV = lambda y: iradon(y, THETA, circle=False, output_size=N).astype(np.float32)

# Low dose CT: adding poisson noise
pht = shepp_logan_phantom()
pht = 0.03 * rescale(pht, scale=512/400, order=0)

prj = A(pht)

i0 = 1e4
dst = np.exp(-prj)
dst = poisson.rvs(i0 * dst)
dst[dst < 1] = 1
dst = -np.log(dst / i0)
dst[dst < 0] = 0

noise = dst - prj

rec = AINV(prj)
rec_noise = AINV(noise)
rec_dst = AINV(dst)

plt.subplot(241)
plt.imshow(pht, cmap='gray', vmin=0, vmax=0.03)
plt.title("Ground Truth")

plt.subplot(242)
plt.imshow(rec, cmap="gray", vmin=0, vmax=0.03)
plt.title("Reconstruction")

plt.subplot(243)
plt.imshow(rec_noise, cmap="gray")
plt.title("Reconstruction using Noise")

plt.subplot(244)
plt.imshow(rec_dst, cmap="gray", vmin=0, vmax=0.03)
plt.title("Reconstruction using Noisy data")

plt.subplot(246)
plt.imshow(prj, cmap="gray")
plt.title("Projection data")

plt.subplot(247)
plt.imshow(noise, cmap="gray")
plt.title("Poisson Noise in projection")

plt.subplot(248)
plt.imshow(dst, cmap="gray")
plt.title("Noisy data")

plt.show()
