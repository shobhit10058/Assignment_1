from math import log2
import random

class data:
	
	def __init__(self) -> None:
		self.attributes = set()
		self.values_of_attributes = {}	
	
	def readByFile(self, input_file: str) -> None:
		
		with open(input_file) as inp:
			self.attributes = list(inp.readline()[:-1].split(','))
			self.values_of_attributes = {}

			for attr in self.attributes:
				self.values_of_attributes[attr] = []
			
			while(1):
				read_str = inp.readline()[:-1]
				if len(read_str) == 0:
					break
				ps_e = list(read_str.split(','))
				indx = 0
				for attr in self.attributes:
					self.values_of_attributes[attr].append(ps_e[indx])
					indx += 1
			self.attributes = set(self.attributes)

	def initAttr(self, attributes: set):
		self.attributes = attributes
		self.values_of_attributes = {attr:[] for attr in (self.attributes)}

	def addExample(self, values: dict):
		for attr in self.attributes:
			self.values_of_attributes[attr].append(values[attr])

	def getExample(self, ind):

		any_attr = self.attributes.pop()
		self.attributes.add(any_attr)
		inst = {}
		for attr in self.attributes:
			inst[attr] = (self.values_of_attributes[attr][ind])
		return inst
		
	def FillMissingVal(self) -> None:
		for attr in (self.attributes):
			co_vals = {}
			ms_val = ""
			ms_co  = 0
			
			for val_ind in range(len(self.values_of_attributes[attr])):
				ps_v = self.values_of_attributes[attr][val_ind]
				if len(ps_v) == 0:
					continue
				if not ps_v in co_vals:
					co_vals[ps_v] = 0
				co_vals[ps_v] += 1
				if ms_co <= co_vals[ps_v]:
					ms_co = co_vals[ps_v]
					ms_val = ps_v
			
			for val_ind in range(len(self.values_of_attributes[attr])):
				ps_v = self.values_of_attributes[attr][val_ind]
				if len(ps_v) == 0:
					self.values_of_attributes[attr][val_ind] = ms_val

	def processValueType(self):
		for attr in self.attributes:
			for indx in range(len(self.values_of_attributes[attr])):
				if attr == 'gender':
					self.values_of_attributes[attr][indx] = (self.values_of_attributes[attr][indx] == 'Female')*1
				else:
					self.values_of_attributes[attr][indx] = float(self.values_of_attributes[attr][indx])

	def split(self, test_frac):
		any_attr = self.attributes.pop()
		self.attributes.add(any_attr)
		data_size = len(self.values_of_attributes[any_attr])
		test_size = int(test_frac * data_size)
		indices = [_ for _ in range(data_size)]
		chosen_test_ind = set(random.sample(indices, test_size))
		train_data = data()
		test_data = data()
		train_data.initAttr(self.attributes)
		test_data.initAttr(self.attributes)
		for ex_ind in range(data_size):
			example = {}
			for attr in (self.attributes):
				example[attr] = (self.values_of_attributes[attr][ex_ind])
			if ex_ind in chosen_test_ind:
				test_data.addExample(example)
			else:
				train_data.addExample(example)
			
		return (train_data, test_data)

class node:

	def __init__(self, target) -> None:
		self.examples = data()
		self.split_attr = 'none'
		self.split_attr_thrs = 'none'
		# list of nodes
		self.children = []
		self.target_attr = target

	def giveAttributes(self, attributes: set) -> None:
		self.examples.initAttr(attributes)
	
	def addExample(self, example: dict) -> None:
		self.examples.addExample(example)

	def giveExamples(self, examples: data) -> None:
		self.examples = examples
	
	def getClassCount(self):
		class_count = {}

		mx_count = 0
		maj = 1
		for entry in self.examples.values_of_attributes[self.target_attr]:
			if not entry in class_count:
				class_count[entry] = 0
			class_count[entry] += 1

			if mx_count < class_count[entry]:
				mx_count = class_count[entry]
				maj = entry
			
		return (class_count, maj)
	
	def splitByThreshold(self, attr, threshold):
		right_child = node(self.target_attr)
		right_child.examples.initAttr(self.examples.attributes)
		left_child = node(self.target_attr)
		left_child.examples.initAttr(self.examples.attributes)
		num_of_exp = len(self.examples.values_of_attributes[attr])

		for ps_ex_indx in range(num_of_exp):
			ps_val = self.examples.values_of_attributes[attr][ps_ex_indx]
			if ps_val > threshold:
				right_child.addExample(self.examples.getExample(ps_ex_indx))
			else:
				left_child.addExample(self.examples.getExample(ps_ex_indx))
		
		return [left_child, right_child]

	def splitNode(self, attr):
		sort_vals = []
		right_child = node(self.target_attr)
		right_child.examples.initAttr(self.examples.attributes)
		num_of_exp = len(self.examples.values_of_attributes[attr])

		for ps_ex_indx in range(num_of_exp):
			ps_val = self.examples.values_of_attributes[attr][ps_ex_indx]
			sort_vals.append([ps_val, ps_ex_indx])
		
		sort_vals.sort()
		mx_gain = 0
		mx_gain_thrsh = -1
		for ind in range(len(sort_vals) - 2, -1, -1):
			thrh = (sort_vals[ind][0] + sort_vals[ind + 1][0]) / 2
			right_child.addExample(self.examples.getExample(sort_vals[ind + 1][1]))
			
			left_child = node(self.target_attr)
			left_child.examples.initAttr(self.examples.attributes)
			for it_ind in range(ind + 1):
				left_child.addExample(self.examples.getExample(sort_vals[it_ind][1]))
			ps_children = [left_child, right_child]
			ps_gain = self.informationGain(ps_children)
			if ps_gain > mx_gain:
				mx_gain = ps_gain
				mx_gain_thrsh = thrh

		return mx_gain_thrsh
	
	def entropy(self):
		class_count, maj = self.getClassCount()
		entropy_value = 0

		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[self.target_attr])
			entropy_value -= (class_count[vals] * log2(class_count[vals]))
		
		return entropy_value

	def ginnnyIndex(self):
		class_count, maj = self.getClassCount()
		giny_indx = 1
		
		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[self.target_attr])
			giny_indx -= (class_count[vals] ** 2)
			
		return giny_indx

	def informationGain(self, cont_child_nodes):
		entropy_change = self.entropy()
		for child in cont_child_nodes:
			prob = len(child.examples.values_of_attributes[self.target_attr]) / len(self.examples.values_of_attributes[self.target_attr])
			entropy_change -= (prob * child.entropy())
		return entropy_change

	def giniGain(self, cont_child_nodes):
		gini_change = self.ginnnyIndex()
		
		for child in cont_child_nodes:
			prob = len(child.examples.values_of_attributes[self.target_attr]) / len(self.examples.values_of_attributes[self.target_attr])
			gini_change -= (prob * child.ginnnyIndex())

		return gini_change

	# assigns the children by splitting with best gain attribute	
	def splitByHeuristic(self, heuristic) -> None:
		# check if pure node
		if(self.entropy() == 0):
			return
		best_gain = 0
		best_attr = "none"
		best_attr_thrs = "none"
		for cont_attr in (self.examples.attributes):
			if cont_attr == self.target_attr:
				continue
			ps_gain = 0
			thrs = self.splitNode(cont_attr)
			ps_split = self.splitByThreshold(cont_attr, thrs)
			if(heuristic == 'information_gain'):
				ps_gain = self.informationGain(ps_split)
			else:
				ps_gain = self.giniGain(ps_split)
			if ps_gain > best_gain:
				best_gain = ps_gain
				best_attr = cont_attr
				best_attr_thrs = thrs

		if(best_attr == ""):
			return	
		
		self.children = self.splitByThreshold(best_attr, best_attr_thrs)
		self.split_attr = best_attr
		self.split_attr_thrs = best_attr_thrs

class DecisionTree:

	def __init__(self, target) -> None:
		self.root = node(target)
		self.target_attr = target

	def recursion_train(self, root, heuristic: str):
		root.splitByHeuristic(heuristic)
		for ch_val in root.children:
			self.recursion_train(root.children[ch_val], heuristic)
	
	def predictInstance(self, example: dict):
		ps_node = self.root
		while(len(ps_node.children) > 0):
			ps_val = example[ps_node.split_attr]
			ps_node = ps_node.children[ps_val > ps_node.split_attr_thrs]
		class_count, maj = ps_node.getClassCount()
		return maj

	# use the value of target attribute stored in -1 index in values of attributes
	# in leaf node at which test data arrives
	def predictTest(self, test: data):
		predictions = []
		for ex_ind in range(len(test.values_of_attributes[self.target_attr])):
			test_inst = {}
			for attr in test.attributes:
				test_inst[attr] = (test.values_of_attributes[attr][ex_ind])
			predictions.append(self.predictInstance(test_inst))
		return predictions

	# implement the following accuracy test
	def test_accuracy(self, test: data) -> float:
		predictions = self.predictTest(test)
		gt_corr = 0
		tot = len(predictions)
		for i in range(tot):
			gt_corr += (predictions[i] == test.values_of_attributes[self.target_attr][i])
		return gt_corr / tot
	
	# add the procedures to add accuracy check at each depth
	def train(self, examples: data, heuristic: str, test: data) -> None:
		# train in bfs format
		self.root.giveExamples(examples)
		queue = [self.root]
		depths = [0]
		i = 0
		f_acc = open(heuristic + "_model_acc_chang" + ".csv", 'w')
		f_acc.write("depth,accuracy\n")
		f = open(heuristic + ".txt", 'w')
		while i < len(queue):
			queue[i].splitByHeuristic(heuristic)
			if i == 0 or depths[i] != depths[i - 1]:
				f.write('\n')
			f.write("[ " + str(queue[i].split_attr) + "," + str(queue[i].split_attr_thrs) + "," + str(queue[i].entropy()) + "," + str(queue[i].examples.values_of_attributes[self.target_attr][0]) + " ]" + "\t")
			for child in (queue[i].children):
				depths.append(depths[i] + 1)
				queue.append(child)
			f_acc.write(str(depths[i]) + ',' + str(self.test_accuracy(test)) + '\n')
			i += 1
		# following lines for testing
		# we can add more checks
		tot = 0
		for i in range(len(queue) - 1, -1, -1):
			if len(queue[i].children) == 0:
				tot += len(queue[i].examples.values_of_attributes[self.target_attr])
		if tot != len(examples.values_of_attributes[self.target_attr]):
			print("Something fishy")
		else:
			print("Tree is proper")

		# uncomment following for recursive training
		# self.recursion_train(self.root, heuristic)

	def prune(self, validation: data):
		run = 0
		org_acc = self.test_accuracy(validation)
		print("starting_pruning")
		while True:
			queue = [self.root]
			i = 0
			org_acc = self.test_accuracy(validation)
			print("validation_acc =",org_acc)
			print("pruning_run =", run)
			mx_acc = 0
			mx_acc_node = self.root
			while i < len(queue):
				ps_ch = list(queue[i].children)
				queue[i].children = []
				ps_acc = self.test_accuracy(validation)
				if ps_acc > mx_acc:
					mx_acc = ps_acc
					mx_acc_node = queue[i]
				queue[i].children = ps_ch
				queue.extend(queue[i].children)
				i += 1
			run += 1
			if mx_acc > org_acc:
				mx_acc_node.split_attr = "none"
				mx_acc_node.split_attr_thrs = 'none'
				mx_acc_node.children = []
			else:
				break
		print("final_validation_acc =",org_acc)

	def printTree(self, file):
		queue = [self.root]
		depths = [0]
		i = 0
		f = open(file, 'w')
		while i < len(queue):
			if i > 0 and depths[i] != depths[i - 1]:
				f.write('\n')
			f.write("[ " + str(queue[i].split_attr) + "," + str(queue[i].split_attr_thrs) + "," + str(queue[i].entropy()) + "," + str(queue[i].getClassCount()[1]) + " ]" + "\t")
			for child in (queue[i].children):
				depths.append(depths[i] + 1)
				queue.append(child)
			i += 1

org_data = data()
org_data.readByFile('data/train.csv')
org_data.FillMissingVal()
org_data.processValueType()

train_data, test_data = org_data.split(0.3)
valid_data, test_data = test_data.split(0.5)

model = DecisionTree('is_patient')
print("starting training with information gain")
model.train(train_data, 'information_gain', test_data)
print("starting_test_acc =", model.test_accuracy(test_data))
model.printTree('starting_tree_inf_gain.txt')
model.prune(valid_data)
print("test_acc =", model.test_accuracy(test_data),'\n')
model.printTree('final_tree_inf_gain.txt')

print("starting training with gini gain")
model.train(train_data, 'gini_gain', test_data)
print("starting_test_acc =", model.test_accuracy(test_data))
model.printTree('starting_tree_gini_gain.txt')
model.prune(valid_data)
print("test_acc =", model.test_accuracy(test_data))
model.printTree('final_tree_gini_gain.txt')
