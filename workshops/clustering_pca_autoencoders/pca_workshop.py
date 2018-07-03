import numpy as np
from PIL import Image

np.random.seed(2339784) # random seed for consistency

#------------------------------------------------------
#      Data setup
#------------------------------------------------------

img = Image.open("Robot2s.png", "r")
data = img.convert("1").getdata()
bytes = np.array(data)
bytes = np.reshape(bytes, (298, 298))

mu_vec1 = np.array([0,0,0])
cov_mat1 = np.array([[50,0,0],[0,50,0],[0,0,50]])
class1_sample = np.random.multivariate_normal(mu_vec1, cov_mat1, 12332).T
count = 0
for i in range(0, 297):
    for j in range (0, 297):
        if bytes[i][j] == 0:
            class1_sample[0][count]  += i
            class1_sample[1][count]  += j
            class1_sample[2][count]  += i
            count = count + 1
assert class1_sample.shape == (3,12332), "The matrix has not the dimensions 3x12332"

mu_vec2 = np.array([200,200,200])
cov_mat2 = np.array([[50,0,0],[0,50,0],[0,0,50]])
class2_sample = np.random.multivariate_normal(mu_vec2, cov_mat2, 12332).T
assert class2_sample.shape == (3,12332), "The matrix has not the dimensions 3x12332"

#------------------------------------------------------
#      Plotting the data
#------------------------------------------------------

from matplotlib.patches import FancyArrowPatch

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

#%pylab inline
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d

fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=1, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=1, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()

all_samples = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_samples.shape == (3,24664), "The matrix has not the dimensions 3x24664"

#------------------------------------------------------
#      Compute the mean
#------------------------------------------------------

mean_x = np.mean(all_samples[0,:])
mean_y = np.mean(all_samples[1,:])
mean_z = np.mean(all_samples[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])

print('Mean Vector:\n', mean_vector)

#------------------------------------------------------
#      TASK 1:  Compute the covariance matrix
#------------------------------------------------------

cov_mat = np.zeros((3,3))
#    COMPUTE THE COVARIANCE MATRIX HERE!!!
print('Covariance Matrix:\n', cov_mat)

#------------------------------------------------------
#      TASK 2: Compute the eigevalue/vector pairs
#------------------------------------------------------

#    COMPUTE THE EIGENVECTORS AND EIGENVALUES FOR THE FROM THE COVARIANCE MATRIX HERE!!!
#    eig_val_cov, eig_vec_cov
for i in range(len(eig_val_cov)):
    print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
    print(40 * '-')

#------------------------------------------------------
#      Plot the eigenvectors
#------------------------------------------------------

fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(111, projection='3d')

sf = 50
ax.plot(all_samples[0,:], all_samples[1,:], all_samples[2,:], 'o', markersize=1, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec_cov.T:
    a = Arrow3D([mean_x, v[0]*sf+mean_x], [mean_y, v[1]*sf+mean_y], [mean_z, v[2]*sf+mean_z], mutation_scale=5, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')

plt.title('Eigenvectors')

plt.show()

#------------------------------------------------------
#      TASK 3: Sort the eigenvectors per the eigenvalue norms
#------------------------------------------------------

#    SORT THE (EIGENVALUE, EIGENVECTOR) TUPLES FROM HIGH TO LOW HERE!

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])

#------------------------------------------------------
#      TASK 4: Compute the projection matrix
#------------------------------------------------------

#    COMPUTE THE PROJECTION MATRIX matrix_w HERE!
print('Matrix W:\n', matrix_w)

#------------------------------------------------------
#      TASK 5: Project onto the new base
#------------------------------------------------------

#     PROJECT THE all_samples TO A MATRIX trensformed HERE!
assert transformed.shape == (2,24664), "The matrix is not 2x24664 dimensional."

#------------------------------------------------------
#      Plot the result
#------------------------------------------------------

plt.plot(transformed[0,0:12332], transformed[1,0:12332], 'o', markersize=2, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,12332:24664], transformed[1,12332:24664], '^', markersize=2, color='red', alpha=0.5, label='class2')
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples step by step approach')
plt.show()