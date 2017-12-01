from sklearn.neural_network import MLPClassifier
import pandas as pd

df = pd.read_csv('Classificaton_RANDOM.csv', header=None)

# setosa and versicolor
y_train = df.iloc[0:206, [3]].values
#print('Expected output:')
#print(y_train.shape)
#print(type(y_train))

# sepal length and petal length
X_train = df.iloc[0:206, [0,1,2]].values
#print('Original data:')
#print(X_train.shape)
#print(type(X_train))

# mlp = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=400, alpha=1e-4,
#                     solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=1000, alpha=1e-2,
                    solver='sgd', verbose=15, tol=1e-4, random_state=1,
                    learning_rate_init=.1)



mlp.fit(X_train, y_train)
print("\nTraining set score: %f" % mlp.score(X_train, y_train))
# print("Test set score: %f" % mlp.score(X_test, y_test))


X_test = df.iloc[207:306, [0,1,2]].values
y_test = df.iloc[207:306, [3]].values
print("Test set score: %f" % mlp.score(X_test, y_test))