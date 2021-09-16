from Data import data
from Decision_Tree import DecisionTree

org_data = data()
org_data.readByFile('data/train.csv')
org_data.FillMissingVal()
org_data.processValueType()

train_data, test_data = org_data.split(0.3)
valid_data, test_data = test_data.split(0.2)

model = DecisionTree('is_patient')
print("starting training with information gain")
model.train(train_data, 'information_gain', test_data, True)
print("starting_test_acc =", model.test_accuracy(test_data))
model.printTree('starting_tree_inf_gain.txt')
model.prune(valid_data)
print("test_acc =", model.test_accuracy(test_data),'\n')
model.printTree('final_tree_inf_gain.txt')

print("starting training with gini gain")
model.train(train_data, 'gini_gain', test_data, True)
print("starting_test_acc =", model.test_accuracy(test_data))
model.printTree('starting_tree_gini_gain.txt')
model.prune(valid_data)
print("test_acc =", model.test_accuracy(test_data))
model.printTree('final_tree_gini_gain.txt')