import sys
import json
import time

from sklearn.externals import joblib

from Retrieval import Retrieval
from Featurizer import Featurizer
from CountFeaturizer import CountFeaturizer
from TfIdfFeaturizer import TfIdfFeaturizer
from Classifier import Classifier
from MultiLayerPerceptron import MultiLayerPerceptron
from MultinomialNaiveBayes import MultinomialNaiveBayes
from SupportVectorMachine import SupportVectorMachine
from Evaluator import Evaluator

class Pipeline(object):
	def __init__(self, trainFilePath, valFilePath, retrievalInstance, featurizerInstance, classifierInstance):
		self.retrievalInstance = retrievalInstance
		self.featurizerInstances = featurizerInstances
		self.classifierInstances = classifierInstances
		trainfile = open(trainFilePath, 'r')
		self.trainData = json.load(trainfile)
		trainfile.close()
		valfile = open(valFilePath, 'r')
		self.valData = json.load(valfile)
		valfile.close()
		self.question_answering()
		outfile = open("resultsTable.html","w")
		outfile.write(self.generate_html(self.result))
		outfile.close()

	def makeXY(self, dataQuestions):
		X = []
		Y = []
		for question in dataQuestions:

			long_snippets = self.retrievalInstance.getLongSnippets(question)
			short_snippets = self.retrievalInstance.getShortSnippets(question)

			X.append(short_snippets)
			Y.append(question['answers'][0])

		return X, Y


	def question_answering(self):
		dataset_type = self.trainData['origin']
		candidate_answers = self.trainData['candidates']
		X_train, Y_train = self.makeXY(self.trainData['questions'][0:10])
		X_val, Y_val_true = self.makeXY(self.valData['questions'])

		self.result = [['Featurizer','Classifier','Accuracy','Precision','Recall','F-Measure']]
		for featurizerInstance in featurizerInstances:
			X_features_train, X_features_val = featurizerInstance.getFeatureRepresentation(X_train, X_val)
			for classifierInstance in classifierInstances:
				#featurization
				clf = classifierInstance.buildClassifier(X_features_train, Y_train)
				#Prediction
				Y_val_pred = clf.predict(X_features_val)
				self.evaluatorInstance = Evaluator()
				a =  self.evaluatorInstance.getAccuracy(Y_val_true, Y_val_pred)
				p,r,f = self.evaluatorInstance.getPRF(Y_val_true, Y_val_pred)
				self.result.append([featurizerInstance.getName(),classifierInstance.getName(),str(a),str(p),str(r),str(f)])

	def generate_html(self,result):
		ts = time.gmtime()
		tsf = time.strftime("%Y-%m-%d %H:%M:%S", ts)
		table = "<html><head><title>QA Pipeline with Learning " + tsf + "</title><link href=\"results.css\" rel=\"stylesheet\"></head><body><h2>" + tsf + " QA Pipeline with Learning</h2>\n<table>\n"
		for i,each_result in enumerate(result):
			table += "<tr>"
			if i == 0 :
				for val in each_result:
					table += "<th>" + val + "</th>"
			else:
				for val in each_result:
					table += "<td>" + val + "</td>"
			table += "</tr>"
		table += "</table>\n</body>\n</html>\n";
		return table

if __name__ == '__main__':
	trainFilePath = sys.argv[1] #please give the path to your reformatted quasar-s json train file
	valFilePath = sys.argv[2] # provide the path to val file
	retrievalInstance = Retrieval()
	featurizerInstances = [CountFeaturizer(),TfIdfFeaturizer()]
	classifierInstances = [MultinomialNaiveBayes(),MultiLayerPerceptron(),SupportVectorMachine()]
	trainInstance = Pipeline(trainFilePath, valFilePath, retrievalInstance, featurizerInstances, classifierInstances)
