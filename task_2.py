from Data import data
from Decision_Tree import DecisionTree

org_data = data()
org_data.readByFile('data/train.csv')
org_data.FillMissingVal()
org_data.processValueType()

acc_inf_gain = 0
acc_gini_gain = 0

for splt_indx in range(10):
	train_data, test_data = org_data.split(0.2)
	print("Training with split number %d" %(splt_indx + 1))
	model = DecisionTree('is_patient')
	model.train(train_data, 'information_gain')
	acc_inf_gain += model.test_accuracy(test_data)

	model.train(train_data, 'gini_gain')
	acc_gini_gain += model.test_accuracy(test_data)

acc_inf_gain /= 10
acc_gini_gain /= 10

print("The average accuracy for information gain is", acc_inf_gain)
print("The average accuracy for gini gain is",acc_gini_gain)