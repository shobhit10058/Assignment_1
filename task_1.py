from data import data
from decision_tree import DecisionTree

train_data = data()
train_data.readByFile('data/train.csv')
train_data.FillMissingVal()
train_data.processValueType()

train_data, test_data = train_data.split(0.2)

model = DecisionTree('is_patient')
model.train(train_data, 'information_gain')
print("test_acc by training with information gain =", model.test_accuracy(test_data))

model.train(train_data, 'gini_gain')
print("test_acc by training with gini gain =", model.test_accuracy(test_data))