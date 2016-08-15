import pandas as pd
from sklearn.decomposition import PCA
from sklearn import manifold

# If you'd like to try this lab with PCA instead of Isomap as the dimensionality reduction technique:
Test_PCA = False


def plotDecisionBoundary(model, X, y):
  print "Plotting..."
  import matplotlib.pyplot as plt
  import matplotlib
  matplotlib.style.use('ggplot') # Look Pretty

  fig = plt.figure()
  ax = fig.add_subplot(111)

  padding = 0.1
  resolution = 0.1

  from matplotlib.colors import ListedColormap
  cmap_light = ListedColormap(['#AAFFAA', '#AAAAFF'])
  cmap_bold  = ListedColormap(['#00AA00', '#0000AA'])
  
  # Calculate the boundaries
  x_min, x_max = X[:, 0].min(), X[:, 0].max()
  y_min, y_max = X[:, 1].min(), X[:, 1].max()
  x_range = x_max - x_min
  y_range = y_max - y_min
  x_min -= x_range * padding
  y_min -= y_range * padding
  x_max += x_range * padding
  y_max += y_range * padding

  # Create a 2D Grid Matrix. The values stored in the matrix are the predictions of the class at said location
  import numpy as np
  xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                       np.arange(y_min, y_max, resolution))

  # What class does the classifier say?
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)

  # Plot the contour map
  plt.contourf(xx, yy, Z, cmap=cmap_light)
  plt.axis('tight')

  # Plot our original points as well...
  for label in np.unique(y):
    #label: (2 for benign, 4 for malignant)
    c = 0 if label==2 else 1
    indices = np.where(y == label)
    plt.scatter(X[indices, 0], X[indices, 1], c=cmap_bold(c), alpha=0.8)

  p = model.get_params()
  plt.title('K = ' + str(p['n_neighbors']))
  plt.show()


# 
# TODO: Load in the dataset into a variable called 'X'.
# Identify nans, and set proper headers.
# Be sure to verify the rows line up by looking at the file in a text editor.
#
# .. your code here ..
X = pd.read_csv('Datasets/breast-cancer-wisconsin.data', names = ['sample', 'thickness', 'size', 'shape', 'adhesion', 'epithelial', 'nuclei', 'chromatin', 'nucleoli', 'mitoses', 'status'])
print X.head()
print X.dtypes # All columns are of type int64, except nuclei, so we need to convert that
X.nuclei = pd.to_numeric(X.nuclei, errors = 'coerce')
print X.isnull().sum() # 16 missing values in nuclei now which need to be filled in
print X.nuclei.unique() # Nuclei takes on the values 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, so filling in using the mean would not be appropriate
X = X.fillna(method = 'backfill', axis = 1)

# 
# TODO: Copy out the status column into a slice, then drop it from the main dataframe. You can also drop the sample column, since that
# doesn't provide us with any machine learning power.
#
# .. your code here ..
y = X.status
X.drop(labels = ['sample', 'status'], axis = 1, inplace = True)

#
# TODO: With the labels safely extracted from the dataset, replace any nan values with the mean feature / column value
#
# .. your code here ..
# No missing/NaN values!

#
# TODO: Experiment with the basic SKLearn preprocessing scalers. We know that the features consist of different units mixed
# in together, so it's reasonable to assume feature scaling is necessary. Print out a description of the dataset, post transformation.
#
# .. your code here ..
from sklearn import preprocessing
T = preprocessing.normalize(X)

#
# PCA and Isomap are your new best friends
model = None
if Test_PCA:
  print "Computing 2D Principal Components"
  #
  # TODO: Implement PCA here. save your model into the variable 'model1'.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  model = PCA(n_components = 2)   

else:
  print "Computing 2D Isomap Manifold"
  #
  # TODO: Implement Isomap here. save your model into the variable 'model1'
  # Experiment with K values from 5-10.
  # You should reduce down to two dimensions.
  #
  # .. your code here ..
  model = manifold.Isomap(n_neighbors = 4, n_components = 2)

#
# TODO: Train your model against X and transform it. You can save the results right back into X itself.
#
# .. your code here ..
X = model.fit_transform(T)

#
# TODO: Do train_test_split. Use the same variable names as on the EdX platform in the reading material, but set the random_state = 7
# for reproducibility, and keep the test_size at 0.33 (33%).
#
# .. your code here ..
from sklearn.cross_validation import train_test_split
data_train, data_test, label_train, label_test = train_test_split(X, y, test_size = 0.33, random_state = 7)

# 
# TODO: Implement and train KNeighborsClassifier on your projected 2D training data here. You can use any K value from 1 - 15,
# so play around with it and see what results you can come up. Your goal is to find a good balance where you aren't too specific
# (low-K), nor are you too general (high-K). You should also experiment with how changing the weights parameter affects the results.
#
# .. your code here ..
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 4, weights = 'distance')
knn.fit(data_train, label_train)

#
# INFO: Be sure to always keep the domain of the problem in mind! It's WAY more important to errantly classify a benign tumor as malignant,
# and have it removed, than to incorrectly leave a malignant tumor, believing it to be benign, and then having the patient progress in
# cancer. Since the UDF weights don't give you any class information, the only way to introduce this data into SKLearn's KNN Classifier
# is by "baking" it into your data. For example, randomly reducing the ratio of benign samples compared to malignant samples from the
# training set.

#
# TODO: Calculate + Print the accuracy of the testing set
#
# .. your code here ..
print knn.score(data_test, label_test)

plotDecisionBoundary(knn, data_test, label_test)

# Lab Question:
# Between the two provided SciKit-Learn options for weighing, which one performed better on this dataset overall, given the many 'K'
# permutations you experimented with?
# Distance