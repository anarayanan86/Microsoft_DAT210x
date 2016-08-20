import pandas as pd
import numpy as np

# Load up the /Module6/Datasets/parkinsons.data data set into a variable X, being sure to drop the name column.
X = pd.read_csv('Datasets/parkinsons.data')
X.drop('name', axis = 1, inplace = True)
print X.head()
print X.info
print X.describe()
print X.isnull().sum() # No NaNs!
print X.dtypes # All object types are correct!

# Splice out the status column into a variable y and delete it from X.
y = X.status
X.drop('status', axis = 1, inplace = True)
print X.columns # 'status' has been dropped from X

# Wait a second. Pull open the dataset's label file from: https://archive.ics.uci.edu/ml/datasets/Parkinsons
# Look at the units on those columns: Hz, %, Abs, dB, etc. What happened to transforming your data? With all of those units
# interacting with one another, some pre-processing is surely in order. Right after you splice out the status column, but before
# you process the train/test split, inject SciKit-Learn pre-processing code. Unless you have a good idea which one is going to work
# best, you're going to have to try the various pre-processors one at a time, checking to see if they improve your predictive accuracy.
# Experiment with Normalizer(), MaxAbsScaler(), MinMaxScaler(), and StandardScaler().
from sklearn import preprocessing

#T = preprocessing.StandardScaler().fit_transform(X)
#T = preprocessing.MinMaxScaler().fit_transform(X)
#T = preprocessing.Normalizer().fit_transform(X)
T = preprocessing.scale(X)
#T = X # No Change

# The accuracy score keeps creeping upwards. Let's have one more go at it. Remember how in a previous lab we discovered that SVM's
# are a bit sensitive to outliers and that just throwing all of our unfiltered, dirty or noisy data at it, particularly in
# high-dimensionality space, can actually cause the accuracy score to suffer?
# Well, let's try to get rid of some useless features. Immediately after you do the pre-processing, run PCA on your dataset. The
# original dataset has 22 columns and 1 label column. So try experimenting with PCA n_component values between 4 and 14. Are you able
# to get a better accuracy?
'''
from sklearn.decomposition import PCA
pca = PCA(n_components = 14)
X_pca = pca.fit_transform(T)
'''
# No, the accuracy levels off at the same value as before from 7 components onwards.

# If you are not, then forget about PCA entirely, unless you want to visualize your data. However if you are able to get a higher score,
# then be *sure* keep that figure in mind, and comment out all the PCA code.
# In the same spot, run Isomap on the data, before sending it to the train / test split. Manually experiment with every inclusive
# combination of n_neighbors between 2 and 5, and n_components between 4 and 6. Are you able to get a better accuracy?
from sklearn.manifold import Isomap

# You're going to have to write nested for loops that wrap around everything from here on down!
best_score = 0
for k in range(2, 6):
    for l in range(4, 7):
        iso = Isomap(n_neighbors = k, n_components = l)
        X_iso = iso.fit_transform(T)

        # Perform a train/test split. 30% test group size, with a random_state equal to 7.
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_iso, y, test_size = 0.3, random_state = 7)

        # Create a SVC classifier. Don't specify any parameters, just leave everything as default.
        # Fit it against your training data and then score your testing data.
        from sklearn.svm import SVC
        # Lines below are for the first lab question:
        '''
        model = SVC()
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        print score
        '''

        # Program a naive, best-parameter searcher by creating a nested for-loops. The outer for-loop should iterate a variable C
        # from 0.05 to 2, using 0.05 unit increments. The inner for-loop should increment a variable gamma from 0.001 to 0.1, using
        # 0.001 unit increments. As you know, Python ranges won't allow for float intervals, so you'll have to do some research on
        # NumPy ARanges, if you don't already know how to use them.

        # Since the goal is to find the parameters that result in the model having the best score, you'll need a best_score = 0 variable
        # that you initialize outside of the for-loops. Inside the for-loop, create a model and pass in the C and gamma parameters into
        # the class constructor. Train and score the model appropriately. If the current best_score is less than the model's score, then
        # update the best_score, being sure to print it out, along with the C and gamma values that resulted in it.        
        for i in np.arange(start = 0.05, stop = 2.05, step = 0.05):
            for j in np.arange(start = 0.001, stop = 0.101, step = 0.001):
                model = SVC(C = i, gamma = j)
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_C = model.C
                    best_gamma = model.gamma
                    best_n_neighbors = iso.n_neighbors
                    best_n_components = iso.n_components
print "The highest score obtained:", best_score
print "C value:", best_C 
print "gamma value:", best_gamma
print "isomap n_neighbors:", best_n_neighbors
print "isomap n_components:", best_n_components

# If you are not, then forget about isomap entirely, unless you want to visualize your data. However if you are able to get a higher
# score, then be *sure* keep that figure in mind.
# If either PCA or Isomap helped you out, then uncomment out the appropriate transformation code so that you have the highest accuracy
# possible.

# Lab Question 1:
# What accuracy did you score?
# 0.813559322034
#
# Lab Question 2:
# After running your assignment again, what is the highest accuracy score you are able to get?
# 0.915254237288
#
# Lab Question 3:
# After trying all of these scalers, what is the new highest accuracy score you're able to achieve?
# 0.932203389831
#
# Lab Question 4:
# What is your highest accuracy score on this assignment to date?
# 0.966101694915
# (C value: 1.7, gamma value: 0.06, isomap n_neighbors: 2, isomap n_components: 4)
