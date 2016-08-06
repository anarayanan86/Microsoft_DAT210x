import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

# Look pretty...
matplotlib.style.use('ggplot')

# Do * NOT * alter this line, until instructed!
scaleFeatures = False

# TODO: Load up the dataset and remove any and all rows that have a nan. You should be a pro at this
# by now ;-)
#
# .. your code here ..
df = pd.read_csv('Datasets/kidney_disease.csv')
df.dropna(axis = 0, how = 'any', inplace = True)
#print df.head()

# Create some color coded labels; the actual label feature will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA
labels = ['red' if i=='ckd' else 'green' for i in df.classification]

# TODO: Use an indexer to select only the following columns:
#       ['bgr','wc','rc']
#
# .. your code here ..
df = df[['bgr', 'wc', 'rc']]

# TODO: Print out and check your dataframe's dtypes. You'll probably want to call 'exit()' after you print it out so you can stop the
# program's execution.
#
# You can either take a look at the dataset webpage in the attribute info section:
# https://archive.ics.uci.edu/ml/datasets/Chronic_Kidney_Disease or you can actually peek through the dataframe by printing a few rows.
# What kind of data type should these three columns be? If Pandas didn't properly detect and convert them to that data type for you,
# then use an appropriate command to coerce these features into the right type.
#
# .. your code here ..
print df
print df.dtypes 	# The columns are all of type "object"
df.bgr = pd.to_numeric(df.bgr)
df.wc = pd.to_numeric(df.wc)
df.rc = pd.to_numeric(df.rc)
print df
print df.dtypes 	# Now they are all floats

# TODO: PCA Operates based on variance. The variable with the greatest variance will dominate. Go ahead and peek into your data using a
# command that will check the variance of every feature in your dataset.
# Print out the results. Also print out the results of running .describe on your dataset.
#
# Hint: If you don't see all three variables: 'bgr','wc' and 'rc', then you probably didn't complete the previous step properly.
#
# .. your code here ..
print df.var()
print "This is the describe output: ", df.describe()

# TODO: This method assumes your dataframe is called df. If it isn't, make the appropriate changes. Don't alter the code in
# scaleFeatures() just yet though!
#
# .. your code adjustment here ..
if scaleFeatures: df = helper.scaleFeatures(df)

# TODO: Run PCA on your dataset and reduce it to 2 components.
# Ensure your PCA instance is saved in a variable called 'pca', and that the results of your transformation are saved in 'T'.
#
# .. your code here ..
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(df)
T = pca.transform(df)

# Plot the transformed data as a scatter plot. Recall that transforming the data will result in a NumPy NDArray. You can either use
# MatPlotLib to graph it directly, or you can convert it to DataFrame and have pandas do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in Module4/assignment1.py, this time we'll convert to a Pandas
# Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we are in P.C. space, so we'll just define the coordinates
# accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c=labels, alpha=0.75, ax=ax)
plt.show()

# Lab Questions:
# Having reviewed the dataset metadata on its website, what are the units of the rc, Red Blood Cell Count feature?
# cells/cumm
#
# Why did the .dropna() method fail to convert all of the columns to an appropriate numeric format?
# There were a few erroneous leading tab / whitespace characters
#
# Sort the features below from the largest to the smallest variance amount.
# wc, bgr, rc
#
# As you know, the first thing PCA does is center your dataset about its mean by subtracting the mean from each sample. Looking at the
# .describe() output of your dataset, particularly the min, max, and mean readings per feature, which feature do you think dominates
# your X axis? How about the Y axis?
# wc dominates the X axis, and bgr dominates the Y axis
#
# According to your labeling, red plots correspond to chronic kidney disease, and green plots are non-CKD patients. Looking at the
# scatter plot, are the two classes completely separable, or are there multiple records mixed together?
# No, a few records are mixed together
#
# Lab Questions (Continued)
#
# Did scaling your features affect their variances at all?
# Yes
#
# After scaling your features, are the green patients without chronic kidney disease more cleanly separable from the red patients
# with chronic kidney disease?
# They are more separable
