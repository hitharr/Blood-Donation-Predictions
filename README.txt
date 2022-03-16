Link to Google Collab: https://colab.research.google.com/drive/1dCSBJwZTMkDqJlGNyaNlcfHpR9N7q_wz
Steps:
Run Cell
	Note: Currently set to use XGBoost and StandardScaler
	In order to test all algorithms, we modified the ### Parameter Variables ### section used at the top of the code

Google Collab code tests the same algorithms and classifiers as python code attached 
	Classifiers: "DecisionTree", "NeuralNetwork", "RandomForest", "AdaBoost", GradinentBoost", "XGBoost"
	Preprocessing Methods: "StandardScaler", "Normalizer", "None"
Additionally:
	Plots roc_auc graph
	Outputs results in the format of [donor id, probability that donor donated in March 2007] and downloads csv file

	
Run from command line:
Python(3) project.py classifier preprocess_method add_features drop_features

Classifiers: "DecisionTree", "NeuralNetwork", "RandomForest", "AdaBoost", GradinentBoost", "XGBoost"

Preprocessing Methods: "StandardScaler", "Normalizer", "None"


Add Features: Binary code 1 means add feature, 0 means do not
Ex: 101 -> add first and third features but not the second
Features:
Donations per month
Time between first and last donation
Average donation

Drop Features: Binary code 1 means add feature, 0 means do not
Ex: 0001 -> only drop fourth default feature
Default features that can be dropped:
Months since last donation
Total number of donations
Months since first donations
Donated blood in March 2007


	
