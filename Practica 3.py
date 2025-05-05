#Imortacion de librerias
from sklearn.neural_network import MLPClassifier
#from sklearn.neural_network import load_iris
from sklearn.datasets import load_iris
data = load_iris()

print(data)
print(len(data['data']))

#Recuperando los datos del data y target
X = data['data']
y = data['target']

print(X[0], y[0])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 30)

print("Tamaño de mi conjunto de entrenamiento: ", len(X_train))
print("Tamaño de mi conjunto de prueba: ", len(X_test))

print("Tamaño de mi conjunto Y prueba: ", len(y_test))
print("Tamaño de mi conjunto Y entrenamieto: ", len(y_train))

# Etapa de entrenamiento
#clf = MLPClassifier() # creando la arquitectura del perceptron multicapa
clf= MLPClassifier(hidden_layer_sizes = (50, 30, 10), activation='tanh', solver='sgd', batch_size = 32, learning_rate = 'adaptive', max_iter = 500)

#Entrenando el MLP
clf.fit(X_train, y_train)

#Realizando una predicción 
prediccion = clf.predict_proba(X_test[:1])
print(prediccion)

predicciones = clf.predict(X_test)
print(predicciones)

#obtener metricas de evaluacion

print(clf.score(X_test, y_test))

from sklearn.metrics import f1_score

print(f1_score(y_test, predicciones, average = 'macro'))

#imprimiendo el reporte de calsificiacion
from sklearn.metrics import classification_report

print("clasificacion reporte ", classification_report(y_test, predicciones))

from sklearn.metrics import confusion_matrix
print("Matriz de confusion\n", confusion_matrix(y_test, predicciones))
