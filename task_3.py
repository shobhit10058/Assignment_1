from Data import data
from Decision_Tree import DecisionTree

train_data = data()
train_data.readByFile('data/train.csv')
train_data.FillMissingVal()
train_data.processValueType()

train_data, test_data = train_data.split(0.2)

model = DecisionTree('is_patient')
model.train(train_data, 'information_gain', test_data, True)