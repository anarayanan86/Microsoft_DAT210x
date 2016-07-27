import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# Look pretty...
matplotlib.style.use('ggplot')


#
# TODO: Load up the Seeds Dataset into a Dataframe
# It's located at 'Datasets/wheat.data'
# 
# .. your code here ..
df = pd.read_csv('Datasets/wheat.data')
print df.head(6)


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the subplot to graph a 3D scatter plot using the area,
# perimeter and asymmetry features. Be sure to use the optional display parameter c='red', and also label your axes.
# 
# .. your code here ..
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel('area')
ax.set_ylabel('perimeter')
ax.set_zlabel('asymmetry')
ax.scatter(df['area'], df['perimeter'], df['asymmetry'], c='red', marker='.')


fig = plt.figure()
#
# TODO: Create a new 3D subplot using fig. Then use the subplot to graph a 3D scatter plot using the width,
# groove and length features. Be sure to use the optional display parameter c='green', and also label your axes.
# 
# .. your code here ..
ax1 = fig.add_subplot(111, projection = '3d')
ax1.set_xlabel('width')
ax1.set_ylabel('groove')
ax1.set_zlabel('asymmetry')
ax1.scatter(df['area'], df['perimeter'], df['asymmetry'], c='green', marker='.')


plt.show()

# Questions:
# Which of the plots seems more compact / less spread out?
# Groove x Length x Width

# Which of the plots were you able to identify two outliers within, that stuck out from the samples?
# Groove x Length x Width