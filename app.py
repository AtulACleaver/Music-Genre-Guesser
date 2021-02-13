import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
"""print(music_data)"""

#     age  gender      genre
# 0    20       1     HipHop
# 1    23       1     HipHop
# 2    25       1     HipHop
# 3    26       1       Jazz
# 4    29       1       Jazz
# 5    30       1       Jazz
# 6    31       1  Classical
# 7    33       1  Classical
# 8    37       1  Classical
# 9    20       0      Dance
# 10   21       0      Dance
# 11   25       0      Dance
# 12   26       0   Acoustic
# 13   27       0   Acoustic
# 14   30       0   Acoustic
# 15   31       0  Classical
# 16   34       0  Classical
# 17   35       0  Classical

# This is our csv file from here we have to divide it into two parts input and output.
# From that I mean that input would be age and gender.
# While the genre will be our output.

# For example we will ask this data set to give us the output for a person of age 21 which is not given in the csv file.
# So, try that.

# Okay so, what I have done here is that I have made a dataset which will not edit the main csv file but will remove the genre
#  section of the file.
x = music_data.drop(columns=['genre'])
# Just to check it.
"""print(x)"""

# This will give us our output.
# This specific method will get only the genre section of the csv file.
y = music_data['genre']
"""print(y)"""


# The next step would be to create an algorithm/ taking it from someone.
# from sklearn.tree import decisionTreeClassifier is one of the most famous algorithms.

model = DecisionTreeClassifier()
model.fit(x, y)

prediction = model.predict([[21, 1], [22, 0]])
print(prediction)
