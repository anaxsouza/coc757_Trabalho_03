import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys


# return the matrix X_approxim = U(m)*S(m)*V(m)t
def get_compress_image(m_dim,n_dim,U, S, Vt, m):
    X_approxim = U[:, :m] @ S[:m, :m] @ Vt[:m, :]

    X_approxim_R = X_approxim[0:m_dim]
    X_approxim_G = X_approxim[m_dim:m_dim*2]
    X_approxim_B = X_approxim[m_dim*2:m_dim*3]

    img_compr = np.zeros((m_dim, n_dim, 3))
    img_compr[..., 0] = X_approxim_R/255
    img_compr[..., 1] = X_approxim_G/255
    img_compr[..., 2] = X_approxim_B/255

    return img_compr


# create figure 1
fig_1 = plt.figure(figsize=(16, 16))
(ax1, ax2) = fig_1.subplots(1, 2)

# original image
img = mpimg.imread('image.jpg')
m_dim = img.shape[0]
n_dim = img.shape[1]

# extract each RGB component, and convert each to a matrix
img_R = img[:, :, 0]
img_G = img[:, :, 1]
img_B = img[:, :, 2]

# create matrix X of size 3MxN, composed by each RGB matrices. The RGB matrices are inserted by row.
X = np.append(img_R, img_G, axis=0)
X = np.append(X, img_B, axis=0)

# calculate the SVD parameters
U, S, Vt = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

# calucate RIC and save in array
ric = np.cumsum(np.diag(S))/np.sum(np.diag(S))

# Adds a subplot at the 1st position
ax1.semilogy(np.diag(S))
ax1.set_xlabel('m', fontsize=12)
ax1.set_ylabel('\u03C3', fontsize=12)
ax1.set_title('Singular values (\u03C3)', fontsize=12)
ax1.grid()

ax2.semilogy(ric)
ax2.set_xlabel('m', fontsize=12)
ax2.set_ylabel('RIC', fontsize=12)
ax2.set_title('Relative Information Content (RIC)', fontsize=12)
ax2.grid()

plt.savefig('figure_1.png')
plt.show()


mode_50 = 0
mode_75 = 0
mode_99 = 0
cond = True

# calculate the modes values to achieve RIC as 0.5, 0.75, 0.99
for i in range(ric.shape[0]):
    if ric[i] >= 0.5:
        mode_50 = i+1
        print(mode_50)
        break
for i in range(ric.shape[0]):
    if ric[i] >= 0.75:
        mode_75 = i+1
        print(mode_75)
        break
for i in range(ric.shape[0]):
    if ric[i] >= 0.99:
        mode_99 = i+1
        print(mode_99)
        break

# compress the images in modes RIC 0.5, 0.75 and 0.99
img_compr_50 = get_compress_image(m_dim,n_dim,U, S, Vt, mode_50)
img_compr_75 = get_compress_image(m_dim,n_dim,U, S, Vt, mode_75)
img_compr_99 = get_compress_image(m_dim,n_dim,U, S, Vt, mode_99)

# create figure 2
fig_2 = plt.figure(figsize=(16, 20))
((ax1, ax2),(ax3,ax4)) = fig_2.subplots(2, 2)

# Adds a subplot at the 1st position
ax1.imshow(img_compr_50)
title_50 = 'RIC=0.5 ; '+ str(mode_50) + ' MODES'
ax1.set_title(title_50,fontsize=12)

# Adds a subplot at the 2nd position
ax2.imshow(img_compr_75)
title_75 = 'RIC=0.75 ; '+ str(mode_75) + ' MODES'
ax2.set_title(title_75,fontsize=12)

# Adds a subplot at the 3rd position
ax3.imshow(img_compr_99)
title_99 = 'RIC=0.99 ; '+ str(mode_99) + ' MODES'
ax3.set_title(title_99,fontsize=12)

# Adds a subplot at the 4th position
ax4.imshow(img)
ax4.set_title('ORIGINAL',fontsize=12)

plt.savefig('figure_2.png')
plt.show()
