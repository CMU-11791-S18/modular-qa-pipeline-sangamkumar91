from Classifier import Classifier
from sklearn.neural_network import MLPClassifier

#This is a subclass that extends the abstract class Classifier.
class MultiLayerPerceptron(Classifier):

	#The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
	def buildClassifier(self, X_features, Y_train):
		clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
		return clf.fit(X_features,Y_train)
