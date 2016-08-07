#
# TODO: Import whatever needs to be imported to make this work
#
# .. your code here ..
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot') # Look Pretty


#
# TODO: To procure the dataset, follow these steps:
# 1. Navigate to: https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2
# 2. In the 'Primary Type' column, click on the 'Menu' button next to the info button,
#    and select 'Filter This Column'. It might take a second for the filter option to
#    show up, since it has to load the entire list first.
# 3. Scroll down to 'GAMBLING'
# 4. Click the light blue 'Export' button next to the 'Filter' button, and select 'Download As CSV'

#
# TODO: Load your dataset after importing Pandas
#
# .. your code here ..
df1 = pd.read_csv('Datasets/Crimes_-_2001_to_present.csv')

#
# TODO: Drop any ROWs with nans in them
#
# .. your code here ..
df1.dropna(axis = 0, how = 'any', inplace = True)

#
# TODO: Print out the dtypes of your dset
#
# .. your code here ..
print df1.dtypes

#
# Coerce the 'Date' feature (which is currently a string object) into real date,
# and confirm by re-printing the dtypes. NOTE: This is a slow process...
#
# .. your code here ..
df1.Date = pd.to_datetime(df1.Date) # Converts the entries in the 'Date' column to datetime64[ns]
print df1.dtypes


def doKMeans(dataframe):

  #
  # TODO: Filter dataframe so that you're only looking at Longitude and Latitude,
  # since the remaining columns aren't really applicable for this purpose.
  #
  # .. your code here ..
  df = pd.concat([dataframe.Longitude, dataframe.Latitude], axis = 1)

  #
  # INFO: Plot your data with a '.' marker, with 0.3 alpha at the Latitude and Longitude locations in your dataset.
  # Longitude = x, Latitude = y!
  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.scatter(x = df.Longitude, y = df.Latitude, marker='.', alpha=0.3, s = 30)

  #
  # TODO: Use K-Means to try and find seven cluster centers in this dataframe.
  #
  # .. your code here ..
  kmeans_model = KMeans(n_clusters = 7, init = 'random', n_init = 60, max_iter = 360, random_state = 43)
  labels = kmeans_model.fit_predict(df)

  #
  # INFO: Print and plot the centroids...
  centroids = kmeans_model.cluster_centers_
  ax.scatter(x = centroids[:,0], y = centroids[:,1], marker='x', c='red', alpha=0.7, linewidths=3, s = 120)
  print centroids


# INFO: Print & Plot your data
doKMeans(df1)

#
# TODO: Filter out the data so that it only contains samples that have a Date > '2011-01-01', using indexing. Then,
# in a new figure, plot the crime incidents, as well as a new K-Means run's centroids.
#
# .. your code here ..
df2 = df1[df1.Date > '2011-01-01']

# INFO: Print & Plot your data
doKMeans(df2)
plt.title("Dates limited to 2011 and later")
plt.show()

# Lab Questions:
# Did your centroid locations change after you limited the date range to +2011?
# Only Slightly...
#
# What about during successive runs of your assignment? Any centroid location changes happened there?
# All clusters have moved but only slightly, and the centroid arrangement still has the same shape for the most part.
