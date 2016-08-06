import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import assignment2_helper as helper

# Look pretty...
matplotlib.style.use('ggplot')


# Do * NOT * alter this line, until instructed!
scaleFeatures = False


# 1. Load up the dataset and drop all the nominal features. Be sure you select the right axis for columns
# and not rows, otherwise Pandas will complain!
#
# .. your code here ..
df = pd.read_csv('Datasets/kidney_disease.csv')

# Create some color coded labels; the actual label feature will be removed prior to executing PCA, since it's unsupervised.
# You're only labeling by color so you can see the effects of PCA.
labels = ['red' if i=='ckd' else 'green' for i in df.classification]


df.drop(labels = ['id', 'classification', 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'], axis = 1, inplace = True)
#print df.head()


# 2. Right after you print out your dataset's dtypes, add an exit() so you can inspect the results. Does everything
# look like it should/properly numeric? If not, make code changes to coerce the remaining column(s).
#print df.dtypes 
#exit()
# The pcv, wc, and rc columns are still of type 'object'.
df.pcv = pd.to_numeric(df.pcv, errors = 'coerce')
df.wc = pd.to_numeric(df.wc, errors = 'coerce')
df.rc = pd.to_numeric(df.rc, errors = 'coerce')
#print df.dtypes # Now everything is floats

# Need to remove the NaN values from the dataframe:
df.dropna(axis = 0, how = 'any', inplace = True)
print df

# TODO: PCA Operates based on variance. The variable with the greatest variance will dominate. Go ahead and peek into your data using
# a command that will check the variance of every feature in your dataset. Print out the results. Also print out the results of
# running .describe on your dataset.
#
# Hint: If you don't see all three variables: 'bgr','wc' and 'rc', then you probably didn't complete the previous step properly.
#
# .. your code here ..
print df.var()
print "This is the describe output: ", df.describe()


# TODO: This method assumes your dataframe is called df. If it isn't, make the appropriate changes. Don't alter the code
# in scaleFeatures() just yet though!
#
# .. your code adjustment here ..
if scaleFeatures: df = helper.scaleFeatures(df)


# TODO: Run PCA on your dataset and reduce it to 2 components. Ensure your PCA instance is saved in a variable called 'pca',
# and that the results of your transformation are saved in 'T'.
#
# .. your code here ..
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
pca.fit(df)
T = pca.transform(df)


# Plot the transformed data as a scatter plot. Recall that transforming the data will result in a NumPy NDArray. You can
# either use MatPlotLib to graph it directly, or you can convert it to DataFrame and have pandas do it for you.
#
# Since we've already demonstrated how to plot directly with MatPlotLib in Module4/assignment1.py, this time we'll convert
# to a Pandas Dataframe.
#
# Since we transformed via PCA, we no longer have column names. We know we are in P.C. space, so we'll just define the
# coordinates accordingly:
ax = helper.drawVectors(T, pca.components_, df.columns.values, plt, scaleFeatures)
T = pd.DataFrame(T)
T.columns = ['component1', 'component2']
T.plot.scatter(x='component1', y='component2', marker='o', c = labels, alpha=0.75, ax=ax)
plt.show()

# Lab Questions:
# After adding in all of numeric columns, do the green, non-chronic kidney disease patients group closer together than before?
# Yes
#
# After converting the nominal features to boolean features, do the green, non-chronic kidney disease patients group even
# closer together than before?
# Yes
