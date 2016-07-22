import pandas as pd


# TODO: Load up the table, and extract the dataset out of it. If you're having issues with this, look
# carefully at the sample code provided in the reading
#
# .. your code here ..
df = pd.read_html('http://espn.go.com/nhl/statistics/player/_/stat/points/sort/points/year/2015/seasontype/2')[0]
#print df

# TODO: Rename the columns so that they match the column definitions provided to you on the website
#
# .. your code here ..
df.columns = ['RK', 'PLAYER', 'TEAM', 'GP', 'G', 'A', 'PTS', '+/-', 'PIM', 'PTS/G', 'SOG', 'PCT', 'GWG', 'PPG', 'PPA', 'SHG', 'SHA']
#print df

# TODO: Get rid of any row that has at least 4 NANs in it
#
# .. your code here ..
df.dropna(thresh = (len(df.columns) - 4), axis = 1)
#print df

# TODO: At this point, look through your dataset by printing it. There probably still are some erroneous rows in there.
# What indexing command(s) can you use to select all rows EXCEPT those rows?
#
# .. your code here ..
df.drop([0, 1, 12, 13, 24, 25, 36, 37], axis = 0, inplace = True)

# TODO: Get rid of the 'RK' column
#
# .. your code here ..
df.drop('RK', axis = 1, inplace = True)
# print df

# TODO: Ensure there are no holes in your index by resetting it. By the way, don't store the original index
#
# .. your code here ..
df.reset_index(inplace = True, drop = True)
print df

# TODO: Check the data type of all columns, and ensure those that should be numeric are numeric
#print df.dtypes
df.GP = pd.to_numeric(df.GP)
df.G = pd.to_numeric(df.G)
df.A = pd.to_numeric(df.A)
df.PTS = pd.to_numeric(df.PTS)
df['+/-'] = pd.to_numeric(df['+/-'])
df.PIM = pd.to_numeric(df.PIM)
df['PTS/G'] = pd.to_numeric(df['PTS/G'])
df.SOG = pd.to_numeric(df.SOG)
df.PCT = pd.to_numeric(df.PCT)
df.GWG = pd.to_numeric(df.GWG)
df.PPG = pd.to_numeric(df.PPG)
df.PPA = pd.to_numeric(df.PPA)
df.SHG = pd.to_numeric(df.SHG)
df.SHA = pd.to_numeric(df.SHA)
print df.dtypes

# TODO: Your dataframe is now ready! Use the appropriate commands to answer the questions on the course lab page.
print "The number of rows in the dataframe is: ", len(df) #40
print "There are", len(df.PCT.unique()), "unique values in the PCT column." #36
print "The value you get by adding the GP values at indices 15 and 16 of this table is", df.ix[15, 'GP'] + df.ix[16, 'GP'] #164
