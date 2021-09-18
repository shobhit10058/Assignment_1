from Data import data
from Decision_Tree import DecisionTree
import matplotlib.pyplot as plt

train_data = data()
train_data.readByFile('data/train.csv')
train_data.FillMissingVal()
train_data.processValueType()

train_data, test_data = train_data.split(0.2)

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
