from Data import data
from Decision_Tree import DecisionTree
import matplotlib.pyplot as plt

org_data = data()
org_data.readByFile('data/train.csv')
org_data.FillMissingVal()
org_data.processValueType()

train_data, test_data = org_data.split(0.2)

model = DecisionTree('is_patient')
# information gain is selected as it performs 
# slightly better as seen in given data
model.train(train_data, 'information_gain')
print("test accuracy before pruning is =", model.test_accuracy(test_data))

max_thr = 0
max_acc = 0
thrs = 0
thresholds = []
train_accs = []
test_accs = []

#40 is chosen by observation
while thrs < 40:
	train_accs.append(model.pruneByChiSquare(thrs))
	ps_acc = (model.test_accuracy(test_data))
	if ps_acc > max_acc:
		max_acc = ps_acc
		max_thr = thrs
	thrs += 0.01
	thresholds.append(thrs)
	test_accs.append(ps_acc)

figure, ax = plt.subplots()
ax.plot(thresholds, test_accs)
ax.plot(thresholds, train_accs)
ax.set_title('test_and_training_accuracy_vs_threshold') 
plt.savefig('test_and_training_accuracy_vs_threshold.png')
plt.show()
plt.close()

# generating the final model
chosen_thrsh = 5.5
train_data, test_data = org_data.split(0.2)
model = DecisionTree('is_patient')
model.train(train_data, 'information_gain')
train_acc = model.pruneByChiSquare(thrs)
print("test accuracy and training accuracy after pruning is =", model.test_accuracy(test_data),"and", train_acc, "respectively")