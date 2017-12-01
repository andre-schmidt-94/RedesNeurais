from sklearn.neural_network import MLPClassifier
import pandas as pd

df = pd.read_csv('Classificaton_Blood.csv', header=None)

# setosa and versicolor
y_train = df.iloc[0:524, [4]].values
#print('Expected output:')
#print(y_train.shape)
#print(type(y_train))

# sepal length and petal length
X_train = df.iloc[0:524, [0,1,2,3]].values
#print('Original data:')
#print(X_train.shape)
#print(type(X_train))

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(10, 5,), max_iter=10000, alpha=1e-6,
                    solver='sgd', verbose=0, tol=1e-8, random_state=1,
                    learning_rate_init=.1)



mlp.fit(X_train, y_train)
print("\nTraining set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))


X_test = df.iloc[525:747, [0,1,2,3]].values
y_test = df.iloc[525:747, [4]].values
print("Test set score: %f" % mlp.score(X_test, y_test))