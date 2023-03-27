import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=278)

def classify_iris(sl, sw, pl, pw):
    if  pl <= 1.9 and pw <= 0.9 and pl >= 0.1 and pw >= 0.1:
        return("setosa")
    elif pl <= 7.0 and pw <= 2.8  and pl >= 4.9 and pw >= 1.5 :
        return("virginica")
    else:
        return("versicolor")
good_predictions = 0
len = test_set.shape[0]
for i in range(len):
    if classify_iris(test_set[i][0],test_set[i][1],test_set[i][2],test_set[i][3]) == test_set[i][4]:
        good_predictions = good_predictions + 1

print(good_predictions)
print(len)
print((good_predictions/len)*100, "%")