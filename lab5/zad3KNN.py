from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

df = pd.read_csv("iris.csv")

(train_set,test_set) = train_test_split(df.values,train_size=0.7,random_state=123)

train_inputs = train_set[:,0:4]
train_classes = train_set[:,4]
test_inputs = test_set[:,0:4]
test_classes = test_set[:,4]

classifier = KNeighborsClassifier(n_neighbors=11)
classifier.fit(train_inputs,train_classes)
pred = classifier.predict(test_inputs)
acc = classifier.score(test_inputs,test_classes)
print(acc)

#train_size=0.7,random_state=123
# n_neighbours=3
#0.9555555555
#n_neighbours=5
#0.9777777777
#n_neighbours=11
#0.97777777777