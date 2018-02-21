from Classifier import Classifier
from sklearn.neural_network import MLPClassifier

#This is a subclass that extends the abstract class Classifier.
class MultiLayerPerceptron(Classifier):

	#The abstract method from the base class is implemeted here to return multinomial naive bayes classifier
	def buildClassifier(self, X_features, Y_train):
		clf = MLPClassifier()
		return clf.fit(X_features,Y_train)
