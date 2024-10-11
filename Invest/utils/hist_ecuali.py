import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

path = os.path.normpath("C:\\Users\\yraul\\OneDrive\Documentos\GitHub\Modelacion3DEntornos\Img2pcd\Data\output\house\depth\IMG_2632_depth.png")
A = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
B = cv2.equalizeHist(A)

#Calcular histograma de la imagen ecualizada
hist_eq = cv2.calcHist([B], [0], None, [255], [0, 255]).flatten()

fig, axs = plt.subplots(2, 2)
plt.tight_layout()
# vmin y vmax para que no aumente el contraste matplotlib
axs[0,0].imshow(A, cmap="gray",vmin=0, vmax=255) 
axs[0,0].axis("off")
axs[0,0].set_title("Original")

axs[0,1].hist(A.flatten(), 255, [0, 255])
axs[0,1].set_title("Histograma Orig")

axs[1,0].imshow(B, cmap="gray",vmin=0, vmax=255)
axs[1,0].set_title("Imagen equalizada")

axs[1,1].hist(B.flatten(), 255, [0, 255])
axs[1,1].set_title("Histograma Equa")

plt.show()