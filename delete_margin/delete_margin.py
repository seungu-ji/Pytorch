import numpy as np 
import matplotlib.pyplot as plt

img = np.random.normal(0, 10, (300, 800))
height, width = img.shape

figsize = (1, height/width) if height>=width else (width/height, 1)
## image with padding and margin
plt.figure(figsize=figsize) 
plt.imshow(img, cmap=plt.cm.Blues)
plt.savefig("./img_with_padding.png", dpi=100)

## image without padding and margin
plt.figure(figsize=figsize) 
plt.imshow(img, cmap=plt.cm.Blues)
plt.axis('off'), plt.xticks([]), plt.yticks([]) # 축 없애기, 틱 없애기
plt.tight_layout() # 현재 figure 상에 있는 공백을 적당히 배치 (필수X)

# left, bottom을 0 =>  경계에 둠
# right, top를 1 => 또 다른 경계에 둠.
# hspace, wspace의 경우는 subplot이 여러 개일때 subplot간의 간격을 의미
plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
plt.savefig("./img_without_padding.png", 
            bbox_inces='tight', 
            pad_inches=0, 
            dpi=100
           )