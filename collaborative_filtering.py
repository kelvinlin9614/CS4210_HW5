#-------------------------------------------------------------------------
# AUTHOR: Zewen Lin
# FILENAME: collaborative_filtering.py
# SPECIFICATION: question5
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2hr
#-----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import operator

df = pd.read_csv('trip_advisor_data.csv', sep=',', header=0) #reading the data by using the Pandas library ()
X_training = np.array(df.values)
true_x = []
for i in X_training:
    temp = []
    for j in i:
        if isinstance(j,float):
            temp.append(j)
    true_x.append(temp)
#iterate over the other 99 users to calculate their similarity with the active user (user 100) according to their category ratings (user-item approach)
   # do this to calculate the similarity:
   #vec1 = np.array([[1,1,0,1,1]])
   #vec2 = np.array([[0,1,0,1,1]])
   #cosine_similarity(vec1, vec2)
   #do not forget to discard the first column (User ID) when calculating the similarities
   #--> add your Python code here
similarity = {}
vec1 = np.array([true_x[len(X_training) - 1]])
for i in range(len(true_x) - 1):
    vec2 = np.array([true_x[i]])
    similarity[i] = cosine_similarity(vec1, vec2)[0][0]

   #find the top 10 similar users to the active user according to the similarity calculated before
   #--> add your Python code here
similarity_sort = sorted(similarity.items(), key=operator.itemgetter(1))
similarity_sort.reverse()

   #Compute a prediction from a weighted combination of selected neighbors’ for both categories evaluated (galleries and restaurants)
   #--> add your Python code here
avg = sum(vec1[0]) / len(vec1[0])
#Gallery
numerator = 0
denominator = 0
col_g = 1
for i in range(len(similarity_sort)):
    numerator += similarity_sort[i][1]*(float(X_training[similarity_sort[i][0]][col_g]) - sum(true_x[i])/len(true_x[i]))
    denominator += similarity_sort[i][1]
result_g = avg + numerator / denominator
print("prediction for Gallery:" + str(result_g))

# Restaurants
# restaurants col in X_training data
numerator = 0
denominator = 0
col_r = 4
for i in range(len(similarity_sort)):
    numerator += similarity_sort[i][1] * (float(X_training[similarity_sort[i][0]][col_r]) - sum(true_x[i])/len(true_x[i]))
    denominator += similarity_sort[i][1]
result_r = avg + numerator/denominator
print("prediction for restaurant: " + str(result_r))





