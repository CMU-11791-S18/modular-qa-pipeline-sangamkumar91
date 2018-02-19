from Featurizer import Featurizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB


#This is a subclass that extends the abstract class Featurizer.
class TfIdfFeaturizer(Featurizer):

	#The abstract method from the base class is implemeted here to return count features
	def getFeatureRepresentation(self, X_train, X_val):
		tfidf_vect = TfidfVectorizer()
		X_train_weight = tfidf_vect.fit_transform(X_train)
		X_val_weight = tfidf_vect.transform(X_val)
		return X_train_weight, X_val_weight
