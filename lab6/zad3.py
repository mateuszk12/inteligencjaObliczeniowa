from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pandas as pd
df = pd.read_csv("./diabetes.csv")
df_norm = df[['pregnant-times','glucose-concentr','blood-pressure','skin-thickness','insulin','mass-index','pedigree-func','age']]
target = df[['class']].replace(['tested_positive','tested_negative'],[0,1])
df = pd.concat([df_norm, target], axis=1)
(train_set,test_set) = train_test_split(df.values,train_size=0.7,random_state=123)
train_inputs = train_set[:,0:8]
train_classes = train_set[:,8]
test_inputs = test_set[:,0:8]
test_classes = test_set[:,8]
scaler = StandardScaler()
scaler.fit(train_inputs)
mlp = MLPClassifier(hidden_layer_sizes=(6,3),max_iter=500,activation="relu")
mlp.fit(train_inputs,train_classes)
predictions_test = mlp.predict(test_inputs)
print(accuracy_score(predictions_test,test_classes))
print(confusion_matrix(predictions_test, test_classes))