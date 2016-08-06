import pandas as pd

from scipy import misc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.pyplot as plt

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Start by creating a regular old, plain, "vanilla" python list. You can call it 'samples'.
#
# .. your code here .. 
samples = []

#
# TODO: Write a for-loop that iterates over the images in the Module4/Datasets/ALOI/32/ folder, appending each of them to
# your list. Each .PNG image should first be loaded into a temporary NDArray, just as shown in the Feature
# Representation reading.
#
# Optional: Resample the image down by a factor of two if you have a slower computer. You can also convert the image from
# 0-255  to  0.0-1.0  if you'd like, but that will have no effect on the algorithm's results.
#
# .. your code here .. 
import os

for file in os.listdir('Datasets/ALOI/32'):
	a = os.path.join('Datasets/ALOI/32', file)
	img = misc.imread(a).reshape(-1)
	samples.append(img)
print len(samples) # 72, as expected since there 72 files in the folder

for file1 in os.listdir('Datasets/ALOI/32i'):	# Also append the 32i images to the list/dataframe
	b = os.path.join('Datasets/ALOI/32i', file1)
	img1 = misc.imread(b).reshape(-1)
	samples.append(img1)

colors = []
for i in range(72):
	colors.append('b')
for j in range(12):
	colors.append('r')


df = pd.DataFrame(samples) # Convert list of numpy arrays to Pandas DataFrame

# Run Isomap on the DataFrame:
from sklearn import manifold
iso = manifold.Isomap(n_neighbors = 6, n_components = 3)
Z = iso.fit_transform(df)

def Plot2D(T, title, x, y):
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y], marker='.', c = colors, alpha=0.7)

def Plot3D(T, title, x, y, z):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection = '3d')
  ax.set_title(title)
  ax.set_xlabel('Component: {0}'.format(x))
  ax.set_ylabel('Component: {0}'.format(y))
  ax.set_zlabel('Component: {0}'.format(z))
  x_size = (max(T[:,x]) - min(T[:,x])) * 0.08
  y_size = (max(T[:,y]) - min(T[:,y])) * 0.08
  z_size = (max(T[:,z]) - min(T[:,z])) * 0.08
  # It also plots the full scatter:
  ax.scatter(T[:,x],T[:,y],T[:,z], marker='.', c = colors, alpha=0.65)

Plot2D(Z, "Isomap transformed data, 2D", 0, 1)
Plot3D(Z, "Isomap transformed data 3D", 0, 1, 2)

# Lab Questions:
#
# Please describe the results of your isomap embedding--either the 3D or 2D one, it doesn't matter:
# The embedding appears to follow an easily traversable, 3D spline The embedding appears to follow an easily traversable, 3D spline
#
# Try reducing the 'n_neighbors' parameter one value at a time. Keep re-running your assignment until the results look visibly
# different. What is the smallest neighborhood size you can have, while maintaining similar manifold embedding results?
# 2


#
# TODO: Once you're done answering the first three questions, right before you converted your list to a dataframe, add in
# additional code which also appends to your list the images in the Module4/Datasets/ALOI/32_i directory. Re-run your
# assignment and answer the final question below.
#
# .. your code here .. 
'''See lines 35-38 above; code is also copied here:
for file1 in os.listdir('Datasets/ALOI/32i'):
	b = os.path.join('Datasets/ALOI/32', file1)
	img1 = misc.imread(b).reshape(-1)
	samples.append(img1)'''

#
# TODO: Convert the list to a dataframe
#
# .. your code here .. 
# See line 39: df = pd.DataFrame(samples)


#
# TODO: Implement Isomap here. Reduce the dataframe df down to three components, using K=6 for your neighborhood size
#
# .. your code here .. 
# See lines 50-52


#
# TODO: Create a 2D Scatter plot to graph your manifold. You can use either 'o' or '.' as your marker. Graph the first two
# isomap components
#
# .. your code here .. 
# See line 78



#
# TODO: Create a 3D Scatter plot to graph your manifold. You can use either 'o' or '.' as your marker:
#
# .. your code here .. 
# See line 79


plt.show()

# Lab Questions:
#
# Reset your 'n_neighbors' if you changed it from 6. After adding in the additional images from the 32_i dataset,
# do examine your 2D and 3D scatter plots again. Have the new samples altered the shape of your original (blue) manifold?
# Only very slightly...
#
# What is the arrangement of the newly added, red samples?
# Isomap rendered the result as a straight line, intersecting the original manifold Isomap rendered the result as a
# straight line, intersecting the original manifold.
