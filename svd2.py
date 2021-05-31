import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def get_compressed(img,m):
    U, S, Vt = np.linalg.svd(img,full_matrices=False)
    S = np.diag(S)
    X_ = U[:,:m]@S[0:m,:m]@Vt[:m,:]
    return X_



# create figure
fig = plt.figure(figsize=(10, 16))
img = mpimg.imread('./trab03/image.jpg')
#print(img)
#implot = plt.imshow(img)

m = 10

img_R = img[:,:,0]
img_G = img[:,:,1]
img_B = img[:,:,2]

R_compr= get_compressed(img_R,m)
G_compr= get_compressed(img_G,m)
B_compr= get_compressed(img_B,m)

img_compr = np.zeros((img.shape[0],img.shape[1],3))#np.dstack((R_compr,G_compr,B_compr))
img_compr[...,0] = R_compr/255
img_compr[...,1] = G_compr/255
img_compr[...,2] = B_compr/255

'''
j = 0

for r in (5,20,100):
    xapprox = U[:,:r]@S[0:r,:r]@Vt[:r,:]
    plt.figure(j+1)
    j+=1
    img = plt.imshow(xapprox)
    plt.show(
'''


# Adds a subplot at the 1st position
fig.add_subplot(1, 2, 1)
# showing image
plt.imshow(img)
plt.axis('off')
plt.title("Original")

# Adds a subplot at the 2nd position
fig.add_subplot(1, 2, 2)
# showing image
plt.imshow(img_compr)
plt.axis('off')
plt.title("Compressed")
print(img_compr)






plt.show()



