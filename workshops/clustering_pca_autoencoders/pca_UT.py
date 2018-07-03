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
ax.plot([0], [0], [0], 'o', markersize=10, color='blue', alpha=0.5, label='class1')
ax.plot([1], [1], [1], '^', markersize=10, alpha=0.5, color='red', label='class2')

plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')

plt.show()

