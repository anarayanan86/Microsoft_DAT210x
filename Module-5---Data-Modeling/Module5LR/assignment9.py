import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

matplotlib.style.use('ggplot') # Look Pretty


def drawLine(model, X_test, y_test, title, R2):
  # This convenience method will take care of plotting your test observations, comparing them to the regression line, and
  # displaying the R2 coefficient
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(X_test, y_test, c = 'g', marker = 'o')
  ax.plot(X_test, model.predict(X_test), color = 'orange', linewidth = 1, alpha = 0.7)

  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_

  plt.show()

def drawPlane(model, X_test, y_test, title, R2):
  # This convenience method will take care of plotting your test observations, comparing them to the regression plane,
  # and displaying the R2 coefficient
  fig = plt.figure()
  ax = Axes3D(fig)
  ax.set_zlabel('prediction')

  # You might have passed in a DataFrame, a Series (slice), an NDArray, or a Python List... so let's keep it simple:
  X_test = np.array(X_test)
  col1 = X_test[:,0]
  col2 = X_test[:,1]

  # Set up a Grid. We could have predicted on the actual col1, col2 values directly; but that would have generated
  # a mesh with WAY too fine a grid, which would have detracted from the visualization
  x_min, x_max = col1.min(), col1.max()
  y_min, y_max = col2.min(), col2.max()
  x = np.arange(x_min, x_max, (x_max-x_min) / 10)
  y = np.arange(y_min, y_max, (y_max-y_min) / 10)
  x, y = np.meshgrid(x, y)

  # Predict based on possible input values that span the domain of the x and y inputs:
  z = model.predict(  np.c_[x.ravel(), y.ravel()]  )
  z = z.reshape(x.shape)

  ax.scatter(col1, col2, y_test, c='g', marker='o')
  ax.plot_wireframe(x, y, z, color='orange', alpha=0.7)

  title += " R2: " + str(R2)
  ax.set_title(title)
  print title
  print "Intercept(s): ", model.intercept_

  plt.show()

#
# INFO: Let's get started!

#
# TODO: First, as is your habit, inspect your dataset in a text editor, or spread sheet application. The first thing you should
# notice is that the first column is both unique (the name of each) college, as well as unlabeled. This is a HINT that it must be the
# index column. If you do not indicate to Pandas that you already have an index column, it'll create one for you, which would be
# undesirable since you already have one.
#
# Review the .read_csv() documentation and discern how to load up a dataframe while indicating which existing column is to be taken
# as an index. Then, load up the College dataset into a variable called X:
#
# .. your code here ..
X = pd.read_csv('Datasets/college.csv', index_col = 0)
print X.head()
print X.info
print X.describe()
print X.dtypes
print X.isnull().sum() # No missing values!
print X.columns

#
# INFO: This line isn't necessary for your purposes; but we'd just like to show you an additional way to encode features directly.
# The .map() method is like .apply(), but instead of taking in a lambda / function, you simply provide a mapping of keys:values.
# If you decide to embark on the "Data Scientist Challenge", this line of code will save you the trouble of converting it through
# other means:
X.Private = X.Private.map({'Yes':1, 'No':0})

#
# TODO: Create your linear regression model here and store it in a variable called 'model'. Don't actually train or do anything else
# with it yet:
#
# .. your code here ..
from sklearn import linear_model
model = linear_model.LinearRegression()

#
# INFO: The first relationship we're interested in is the amount charged for room and board, as a function of the number of
# accepted students.

#
# TODO: Using indexing, create two slices (series). One will just store the room and board column, the other will store the accepted
# students column. Then use train_test_split to cut your data up into X_train, X_test, y_train, y_test, with a test_size of 30% and
# a random_state of 7.
#
# Since the objective is to model the amount charged for room and board as a function() of the number of accepted students, it should
# be clear to you that your output will be the room and board amount, and your input will be the accepted students amount.
#
# .. your code here ..
s1 = X.Accept
s2 = X[['Room.Board']]
print type(s1), type(s2) # Remember train_test_split can only handle DataFrames, not Series!
s1 = s1.to_frame()

from sklearn.cross_validation import train_test_split
X_train1, X_test1, y_train1, y_test1 = train_test_split(s1, s2, test_size = 0.3, random_state = 7)

#
# TODO: Fit and score your model appropriately. Store the score in the score variable.
#
# .. your code here ..
model1 = model.fit(X_train1, y_train1)
score1 = model1.score(X_test1, y_test1)

# INFO: We'll take it from here, buddy:
drawLine(model1, X_test1, y_test1, "Accept(Room&Board)", score1)

#
# TODO: Duplicate the process above; this time, model the number of enrolled students per college, as a function of the number of
# accepted students
#
# .. your code here ..
s3 = X.Enroll.to_frame()
X_train2, X_test2, y_train2, y_test2 = train_test_split(s1, s3, test_size = 0.3, random_state = 7)
model2 = model.fit(X_train2, y_train2)
score2 = model2.score(X_test2, y_test2)
drawLine(model2, X_test2, y_test2, "Accept(Enroll)", score2)

#
# TODO: Duplicate the process above; this time, model the number of failed undergraduate students per college, as a function of
# the number of accepted students
#
# .. your code here ..
s4 = X[['F.Undergrad']]
X_train3, X_test3, y_train3, y_test3 = train_test_split(s1, s4, test_size = 0.3, random_state = 7)
model3 = model.fit(X_train3, y_train3)
score3 = model3.score(X_test3, y_test3)
drawLine(model3, X_test3, y_test3, "Accept(F.Undergrad)", score3)

#
# TODO: Duplicate the process above (almost). This time is going to be a bit more complicated. Instead of modeling one feature as
# a function of another, you will attempt to do multivariate linear regression to model one feature as a function of TWO other
# features.
#
# Model the number of accepted students, as a function of the amount charged for room and board, AND the number of enrolled students.
# To do this, instead of creating a regular slice for a single-feature input, simply create a slice that contains both columns you
# wish to use as inputs. Your training labels will remain a single slice.
#
# .. your code here ..
s5 = X[['Room.Board', 'Enroll']]
X_train4, X_test4, y_train4, y_test4 = train_test_split(s5, s1, test_size = 0.3, random_state = 7)
model4 = model.fit(X_train4, y_train4)
score4 = model4.score(X_test4, y_test4)
drawPlane(model4, X_test4, y_test4, "Accept(Room&Board,Enroll)", score4)

#
# INFO: That concludes this assignment
#

# INFO + HINT On Fitting, Scoring, and Predicting:
#
# Here's a hint to help you complete the assignment without pulling your hair out! When you use .fit(), .score(), and .predict() on
# your model, SciKit-Learn expects your training data to be in spreadsheet (2D Array-Like) form. This means you can't simply
# pass in a 1D Array (slice) and get away with it.
#
# To properly prep your data, you have to pass in a 2D Numpy Array, or a dataframe. But what happens if you really only want to pass
# in a single feature?
#
# If you slice your dataframe using df[['ColumnName']] syntax, the result that comes back is actually a *dataframe*. Go ahead and do
# a type() on it to check it out. Since it's already a dataframe, you're good -- no further changes needed.
#
# But if you slice your dataframe using the df.ColumnName syntax, OR if you call df['ColumnName'], the result that comes back is
# actually a series (1D Array)! This will cause SKLearn to bug out. So if you are slicing using either of those two techniques, before
# sending your training or testing data to .fit / .score, do a my_column = my_column.reshape(-1,1). This will convert your 1D
# array of [n_samples], to a 2D array shaped like [n_samples, 1]. A single feature, with many samples.
#
# If you did something like my_column = [my_column], that would produce an array in the shape of [1, n_samples], which is incorrect
# because SKLearn expects your data to be arranged as [n_samples, n_features]. Keep in mind, all of the above only relates to your
# "X" or input data, and does not apply to your "y" or labels.


#
# Data Scientist Challenge
# ========================
#
# You've experimented with a number of feature scaling techniques already, such as MaxAbsScaler, MinMaxScaler, Normalizer,
# StandardScaler and more from http://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing
#
# What happens if you apply scaling to your data before doing linear regression? Would it alter the quality of your results?
# Do the scalers that work on a per-feature basis, such as MinMaxScaler behave differently that those that work on a multi-feature
# basis, such as normalize? And moreover, once your features have been scaled, you won't be able to use the resulting regression
# directly... unless you're able to .inverse_transform() the scaling. Do all of the SciKit-Learn scalers support that?
#
# This is your time to shine and to show how much of an explorer you are: Dive deeper into uncharted lands, browse SciKit-Learn's
# documentation, scour Google, ask questions on Quora, Stack-Overflow, and the course message board, and see if you can discover
# something that will be of benefit to you in the future!

# Lab Question:
# Which two relationship had the worst R2 correlations?
# Accept[Room.Board] and Accept[F.Undergrad] Accept[Room.Board] and Accept[F.Undergrad] - correct
