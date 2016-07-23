import pandas as pd
import numpy as np


# TODO:
# Load up the dataset, setting correct header labels
# Use basic pandas commands to look through the dataset...
# get a feel for it before proceeding!
# Find out what value the dataset creators used to represent "nan" and ensure it's properly encoded as np.nan
#
# .. your code here ..
df = pd.read_csv('Datasets/census.data', names = ['education', 'age', 'capital-gain', 'race', 'capital-loss', 'hours-per-week', 'sex', 'classification'])
df[df == 0] = np.nan
#print df.head(15)

# TODO:
# Figure out which features should be continuous + numeric
# Convert these to the appropriate data type as needed, that is, float64 or int64
#
# .. your code here ..
print df.dtypes
# No conversion required!

# TODO:
# Look through your data and identify any potential categorical features. Ensure you properly encode any ordinal types using
# the method discussed in the chapter.
#
# .. your code here ..

# Categorial ordinal feature:
print df.education.unique()
# education categories: ['Bachelors' 'HS-grad' '11th' 'Masters' '9th' 'Some-college' '7th-8th' 'Doctorate' '5th-6th' '10th' '1st-4th' 'Preschool' '12th']
education_ordered = ['Preschool', '1st-4th', '5th-6th', '7th-8th', '9th', '10th', '11th', '12th', 'HS-grad', 'Some-college', 'Bachelors', 'Masters', 'Doctorate']
df.education = df.education.astype("category", ordered = True, categories = education_ordered).cat.codes
#print df

# TODO:
# Look through your data and identify any potential categorical features. Ensure you properly encode any nominal types by
# exploding them out to new, separate, boolean features.
#
# .. your code here ..

# Categorial nominal features below (these are nominal becuase there is no ordering here):
#print df.race.unique()
# race categories: ['White' 'Black' 'Asian-Pac-Islander' 'Amer-Indian-Eskimo' 'Other']
df = pd.get_dummies(df, columns=['race'])
print df.sex.unique()
# sex categories: ['Male' 'Female']
df = pd.get_dummies(df, columns=['sex'])
print df.classification.unique()
# classification categories: ['<=50K' '>50K']
df = pd.get_dummies(df, columns=['classification'])

# TODO:
# Print out your dataframe
print df
print "The number of columns now in the dataframe:", len(df.columns)
# 1 column (education) is a categorial ordinal variable
# 9 boolean columns were created, total
# The dataset is now 14 columns wide
