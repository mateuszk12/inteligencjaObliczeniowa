from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pandas as pd

df = pd.read_csv("iris.csv")

(train_set,test_set) = train_test_split(df.values,train_size=0.7,random_state=123)

train_inputs = train_set[:,0:4]
train_classes = train_set[:,4]
test_inputs = test_set[:,0:4]
test_classes = test_set[:,4]

gnb = GaussianNB()
gnb.fit(train_inputs,train_classes)
pred = gnb.predict(test_inputs)
accuracy = accuracy_score(pred,test_classes)
print(accuracy)

#0.9555555556