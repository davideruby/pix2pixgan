import torch
import torchvision
import torchvision.transforms as transforms
import PIL
import matplotlib.pyplot as plt

im = PIL.Image.open('doge.jpg')
T = transforms.Compose([
    transforms.CenterCrop((256, 256)),
    transforms.ToTensor()
])

x = T(im) #shape 3,256,256
x = x.unsqueeze(0) #shape 1,3,256,256

b, c, h, w = x.shape
k = 32 #kernel_size: 32x32 patches
stride = 32 #stride: 8 patches per row/col (8x8 tot patches)

# per dividere immagine originale (256x256) in patches da 32x32
patches = x.unfold(2, k, stride)
print(patches.shape)
patches = patches.unfold(3, k, stride)
print(patches.shape)

patches = patches.reshape(b, c, -1, k, k) #dim: (batch_size, channels, num_patches, w, h)
patches = patches.permute(0, 2, 1, 3, 4) #dim: (batch_size, num_patches, channels, w, h)
num_patches = patches.shape[1]


patches = patches.reshape(-1, c, k, k) #dim: (batch_size*num_patches, channels, w, h)
print(patches.shape)

grid = torchvision.utils.make_grid(patches)
plt.imshow(grid.permute(1, 2, 0))
plt.show()

# per ricostruire immagine originale partendo dalle patches
# (o dagli output del modello)
output_channels = 3  #change
patches = patches.reshape(b, num_patches, output_channels, k, k)
patches = patches.reshape(b, num_patches, c*k*k).permute(0, 2, 1) #num_patches deve essere ultimo
patches = torch.nn.functional.fold(patches, (h, w), k, stride=k)
print(patches.shape)  #1x3x256x256 -> come originale gg

plt.imshow(patches[0].numpy().transpose(1, 2, 0))
plt.show()