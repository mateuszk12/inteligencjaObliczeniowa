import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
df = pd.read_csv("iris.csv")
(train_set, test_set) = train_test_split(df.values, train_size=0.7,
random_state=278)
train_inputs = train_set[:, 0:4]
train_classes = train_set[:, 4]
test_inputs = test_set[:, 0:4]
test_classes = test_set[:, 4]
dtc = DecisionTreeClassifier()
dtc.fit(train_inputs,train_classes)
test_predict_prob = dtc.predict(test_inputs)
tree.plot_tree(dtc)
print(dtc.score(test_inputs,test_classes)*100,"%")
print(confusion_matrix(test_classes,test_predict_prob))
plt.show()
