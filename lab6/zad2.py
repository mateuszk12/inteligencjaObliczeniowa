from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
iris = load_iris()
datasets = train_test_split(iris.data,iris.target,test_size=0.7)

train_data,test_data,train_labels,test_labels = datasets

scaler = StandardScaler()
scaler.fit(train_data)

train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

mlp = MLPClassifier(hidden_layer_sizes=(3,3),max_iter=2000)
mlp.fit(train_data,train_labels)
predictions_test = mlp.predict(test_data)
print(accuracy_score(predictions_test,test_labels))
#2
#0.7904761904761904
#3
#0.8952380952380953
#3,3
#0.9428571428571428