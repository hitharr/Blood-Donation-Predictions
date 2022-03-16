import numpy as np
import pandas as pd

from sklearn import preprocessing

# From model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# From metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

# Classifiers
from sklearn import tree
from sklearn import neural_network
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

import sys


def main():
	if len(sys.argv) != 5:
		print('ERROR: Incorrect Format')
		return
		
	if len(sys.argv[3]) != 3:
		print('ERROR: Incorrect Format. There are 3 features that can be added.')
		return
		
	if len(sys.argv[4]) != 4:
		print('ERROR: Incorrect Format. There are 4 features that can be dropped.')
		return
	
	### Parameter Variables ###
	test_size = 0.2
	random_state = 0
	classifier = sys.argv[1]
	preproc = sys.argv[2]

	### Pre-Processing ###

	# Get data
	names = ['Recency', 'Frequency', 'Monetary', 'Time', 'Class']
	dataFrame = pd.read_csv('data/transfusion.data.txt')

	# Change column headers from their original values from the repository
	dataFrame.columns = ['Months since last donation', 'Total number of donations', 'Total volume of blood donated', 'Months since first donations', 'Donated blood in March 2007']
	#print(dataFrame)


	# Add Features #

	# Donations per month
	if sys.argv[3][0] == "1":
		dataFrame['Donations per month'] = dataFrame['Total number of donations'] / dataFrame['Months since first donations'];
	# Time between first and last donation
	if sys.argv[3][1] == "1":
		dataFrame['Time as a donar'] =  dataFrame['Months since first donations'] - dataFrame['Months since last donation'];
		# Average donation
	if sys.argv[3][2] == "1":
		dataFrame['Average donation'] = dataFrame['Total volume of blood donated'] / dataFrame['Total number of donations'];


	# Drop feature(s) #
	if sys.argv[4][0] == "1":
		dataFrame = dataFrame.drop(['Months since last donation'], 1)
	if sys.argv[4][1] == "1":
		dataFrame = dataFrame.drop(['Months since first donations'], 1)
	if sys.argv[4][2] == "1":
		dataFrame = dataFrame.drop(['Total number of donations'], 1)
	if sys.argv[4][3] == "1":
		dataFrame = dataFrame.drop(['Total volume of blood donated'], 1)

	# Set up X and y #

	# Get X and y
	X = np.array(dataFrame.drop(['Donated blood in March 2007'], 1))
	y = np.array(dataFrame['Donated blood in March 2007'])
	#print(X)
	#print(y)

	# Scale X 
	if preproc == "StandardScaler":
		scaler = preprocessing.StandardScaler()
		scaler.fit(X)
		preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
	elif preproc == "Normalizer":
		scaler = preprocessing.Normalizer()
		scaler.fit(X)
	

	# Split the dataset #
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)


	### Perform Classification ###

	# Set tuned parameters for classifier  #
	if classifier == "DecisionTree":
		tuned_parameters = [{'max_depth': [1, 3, 5, 10, 20, 50, 100], 'max_leaf_nodes': [None, 2, 3, 5, 10, 20, 50, 100], 'max_features': [None,'sqrt','log2'], 'min_samples_split': [2, 5, 10 , 25, 50, 75, 100], 'min_samples_leaf': [1, 2, 3, 5, 10]}]

	elif classifier == "NeuralNetwork":
		tuned_parameters = [{'activation': ['identity', 'logistic', 'tanh', 'relu'], 'hidden_layer_sizes': [1,10,50,100,200], 'alpha': [.0001,.0005,.001], 'early_stopping': [True, False]}, {'learning_rate': ['adaptive', 'constant']}]

	elif classifier == "RandomForest":
		tuned_parameters = [{'n_estimators': [1,5,10,20],'criterion': ['gini', 'entropy'], 
						'max_features': [1,2,3,4], 'max_depth': [5,10, 15, 20, 25, 50, 75, 100]}
						]
		
	elif classifier == "AdaBoost":
		tuned_parameters = [{'n_estimators': [1,5,10,20],'learning_rate': [.001, .01, .005, .05], 
							'algorithm': ['SAMME','SAMME.R'], 'random_state': [1, 5,10, 15, 20]}
							]	
	elif classifier == "GradientBoost":
		tuned_parameters = [{'n_estimators': [1,5,10,20, 50],'learning_rate': [.001, .01, .005, .05], 
							'min_samples_leaf': [5,10, 15, 20], 'min_samples_split': [5,10, 15, 20, 25, 50]}
							]
		
	elif classifier == "XGBoost":
		tuned_parameters = [{'learning_rate': [1, .5, 2], 'n_estimators': [10, 20, 50, 100, 200], 'booster':['gbtree', 'gblinear', 'dart'], 'max_delta_step': [0,  .5, 1, 5, 10]}]
		
	else:
		print('ERROR: Incorrect Classifier')
		return

	# Scores to use when evaluating classification results
	scores = ["accuracy"]

	# Evaluate classifiers for each score
	for score in scores:
		print("# Tuning hyper-parameters for %s" % score)
		print()

		# Set classifier #
		if classifier == "DecisionTree":
			clf = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='%s' % score)
		elif classifier == "NeuralNetwork":
			clf = GridSearchCV(neural_network.MLPClassifier(), tuned_parameters, cv=5, scoring='%s' % score)
		elif classifier == "RandomForest":
			clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='%s' % score)
		elif classifier == "AdaBoost":
			clf = GridSearchCV(AdaBoostClassifier(), tuned_parameters, cv=5, scoring='%s' % score)
		elif classifier == "GradientBoost":
			clf = GridSearchCV(GradientBoostingClassifier(), tuned_parameters, cv=5, scoring='%s' % score)
		elif classifier == "XGBoost":
			clf = GridSearchCV(XGBClassifier(), tuned_parameters, cv=5, scoring='%s' % score)
		else:
			print('ERROR: Incorrect Classifier')
			return
			
		# Fit the data and print the results #
		clf.fit(X_train, y_train)

		print("Best parameters set found on development set:")
		print()
		print(clf.best_params_)
		print()
		print("Grid scores on development set:")
		print()
		means = clf.cv_results_['mean_test_score']
		stds = clf.cv_results_['std_test_score']
		for mean, std, params in zip(means, stds, clf.cv_results_['params']):
			print("%0.3f (+/-%0.03f) for %r"
				  % (mean, std * 2, params))
		print()

		print("Detailed classification report:")
		print()
		print("The model is trained on the full development set.")
		print("The scores are computed on the full evaluation set.")
		print()
		y_true, y_pred = y_test, clf.predict(X_test)
		print(classification_report(y_true, y_pred))
		print("Detailed confusion matrix:")
		print(confusion_matrix(y_true, y_pred))
		print("Accuracy Score: \n")
		print(accuracy_score(y_true, y_pred))
		print()
		
		
		probability = clf.predict_proba(X_test)
		print("Probability")
		print(probability)
		
		logloss = log_loss(y_true, probability)
		print("Logloss")
		print(logloss)


main()