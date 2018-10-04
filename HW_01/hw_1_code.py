from sklearn.tree import DecisionTreeClassifier
import numpy
import matplotlib.pyplot as plt
from numpy import genfromtxt

#Features Extracted From .data and .csv files.
def data_to_numpy(file, num_features):
	
	if "wilt" in file: 
		#wilt is in .csv, hence we use numpy.genfromtxt which loads csv to a numpy array faster and easily.
		X = genfromtxt(file, delimiter=',')
		print(X.shape)
		return X
	
	else:
	
		X = numpy.empty([0,num_features])
	
		with open(file, 'r') as f:
			content = f.readlines()

			for line in content:
				row = numpy.empty(0)

				for x in line.strip("\n,\'").split():
					row = numpy.append(row, float(x))
				
				X = numpy.append(X, [row], 0)
			print(X.shape)

		return X
#Labels Extracted From .label files.
def label_to_numpy(file):
	
	y = numpy.empty(0)
	
	with open(file, 'r') as f:
		content = f.readlines()
	
		for line in content:
			y = numpy.append(y, int(line))
	
		print(y.shape)
	
	return y
			
datasets_training_features = [["MADELON/madelon_train.data", 500],
							  ["Gisette/gisette_train.data", 5000],
							  ["wilt/wilt_train.csv", 6]]
	

datasets_test_features = [["MADELON/madelon_valid.data", 500],
						  ["Gisette/gisette_valid.data", 5000],
						  ["wilt/wilt_test.csv", 6]]

datasets_training_labels = ["MADELON/madelon_train.labels",
							"Gisette/gisette_train.labels",
							"wilt/wilt_train.labels"]

datasets_test_labels = ["MADELON/madelon_valid.labels",
						"Gisette/gisette_valid.labels",
						"wilt/wilt_test.labels"]

for i in range(0,len(datasets_test_labels)):
#Get the Training and Validation Sets 		
	X_train = data_to_numpy(datasets_training_features[i][0], datasets_training_features[i][1])
	y_train = label_to_numpy(datasets_training_labels[i])

	X_test = data_to_numpy(datasets_test_features[i][0], datasets_test_features[i][1])
	y_test = label_to_numpy(datasets_test_labels[i])


	#Initialise empty arrays to store the mis-classification errors on training and testing sets
	scores_train = numpy.empty(0)
	scores_test = numpy.empty(0)

	#The depths of DT are all taken in a list of numbers as given below:
	depth_list = [1,2,3,4,5,6,7,8,9,10,11,12]

	for depth in depth_list:
		#We used DecisionTreeClassifier from sklearn.
		model = DecisionTreeClassifier(max_depth=depth, random_state=0, min_samples_leaf=1, max_leaf_nodes=datasets_test_features[i][1], presort=True)
		model.fit(X_train, y_train)

		#Gives the accuracy of the model on the training/testing set (above/below). The miscalculation error is computed as (1-accuracy)*100.00%
		training_scores = model.score(X_train, y_train)
		test_scores = model.score(X_test, y_test)
		
		#Error for DT with depth 1 will be at index 0. scores_train[0] gives efficiency of DT with depth 1. Also, remember scores is accuracy 
		scores_train = numpy.append(scores_train, (1 - training_scores)*100.00)
		scores_test = numpy.append(scores_test, (1 - test_scores)*100.00)

	print(scores_test)
	
	minimum_depth = numpy.argmin(scores_test)
	
	#Because of the comment above, we add 1 to the minimum_depth, as the latter is the INDEX of the numpy array with lowest value (See numpy.argmin on google)
	print("Minimum test classification error was achieved on tree with depth: ", minimum_depth+1, " and the minimum test classification error was: ", scores_test[minimum_depth])
	
	#Plotting the graph
	fig = plt.figure()
	plt.plot(depth_list, scores_train, label='Training Validation')
	plt.plot(depth_list, scores_test, label='Testing Validation')
	plt.xlabel('Depth of Desicion Tree')
	plt.ylabel('Mis-classification Error')
	plt.title("Mis-classification Error vs Desicion Tree Depth for " + str(datasets_test_labels[i].split('/')[0]) + " Dataset")
	plt.legend()
	fig.savefig("hw1_"+str(datasets_test_labels[i].split('/')[0])+".png")


