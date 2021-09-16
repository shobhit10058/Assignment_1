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
		self.split_attr = ""
		# list of nodes
		self.children = {}
		self.target_attr = target

	def giveAttributes(self, attributes: set) -> None:
		self.examples.initAttr(attributes)
	
	def addExample(self, example: dict) -> None:
		self.examples.addExample(example)

	def giveExamples(self, examples: data) -> None:
		self.examples = examples
	
	def getClassCount(self):
		class_count = {}

		for entry in self.examples.values_of_attributes[self.target_attr]:
			if not entry in class_count:
				class_count[entry] = 0
			class_count[entry] += 1

		return class_count

	def splitNode(self, attr):
		map_val_to_children = {}

		ps_attrs = set(self.examples.attributes)
		ps_attrs.remove(attr)
		
		for ps_ex_indx in range(len(self.examples.values_of_attributes[attr])):
			ps_val = self.examples.values_of_attributes[attr][ps_ex_indx]
			
			if not ps_val in map_val_to_children:
				map_val_to_children[ps_val] = node(self.target_attr)
				map_val_to_children[ps_val].giveAttributes(ps_attrs)
			
			ps_example = {}
			
			for ps_attr in self.examples.attributes:
				if ps_attr != attr:
					ps_example[ps_attr] = self.examples.values_of_attributes[ps_attr][ps_ex_indx]

			map_val_to_children[ps_val].addExample(ps_example)

		return map_val_to_children
	
	def entropy(self):
		class_count = self.getClassCount()
		entropy_value = 0

		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[self.target_attr])
			entropy_value -= (class_count[vals] * log2(class_count[vals]))
		
		return entropy_value

	def ginnnyIndex(self):
		class_count = self.getClassCount()
		giny_indx = 1
		
		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[self.target_attr])
			giny_indx -= (class_count[vals] ** 2)
			
		return giny_indx

	def informationGain(self, attr):
		entropy_change = self.entropy()
		cont_child_nodes = self.splitNode(attr)
		
		for val in cont_child_nodes:
			prob = len(cont_child_nodes[val].examples.values_of_attributes[self.target_attr]) / len(self.examples.values_of_attributes[self.target_attr])
			entropy_change -= (prob * cont_child_nodes[val].entropy())

		return entropy_change

	def giniGain(self, attr):
		gini_change = self.ginnnyIndex()
		cont_child_nodes = self.splitNode(attr)
		
		for val in cont_child_nodes:
			prob = len(cont_child_nodes[val].examples.values_of_attributes[self.target_attr]) / len(self.examples.values_of_attributes[self.target_attr])
			gini_change -= (prob * cont_child_nodes[val].ginnnyIndex())

		return gini_change

	# assigns the children by splitting with best gain attribute	
	def splitByHeuristic(self, heuristic) -> None:
		# check if pure node
		if(self.entropy() == 0):
			return
		best_gain = 0
		best_attr = ""

		for cont_attr in (self.examples.attributes):
			if cont_attr == self.target_attr:
				continue
			ps_gain = 0
			if(heuristic == 'information_gain'):
				ps_gain = self.informationGain(cont_attr)
			else:
				ps_gain = self.giniGain(cont_attr)
			
			if ps_gain > best_gain:
				best_gain = ps_gain
				best_attr = cont_attr
			
		if(best_attr == ""):
			return	
		
		self.children = self.splitNode(best_attr)
		self.split_attr = best_attr

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
			if not ps_val in ps_node.children:
				break
			ps_node = ps_node.children[ps_val]
		class_count = ps_node.getClassCount()
		count = 0
		pred = 0
		for val in class_count:
			if count < class_count[val]:
				count = class_count[val]
				pred = val
		return pred

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
		values = [0]
		i = 0
		f_acc = open(heuristic + "_model_acc_chang" + ".csv", 'w')
		f_acc.write("depth,accuracy\n")
		f = open(heuristic + ".txt", 'w')
		while i < len(queue):
			queue[i].splitByHeuristic(heuristic)
			if i == 0 or depths[i] != depths[i - 1]:
				f.write('\n')
			f.write(str(queue[i].split_attr) + "," + str(values[i]) + "," + str(queue[i].entropy()) + "," + str(queue[i].examples.values_of_attributes[self.target_attr][0]) + "\t")
			for val in (queue[i].children):
				depths.append(depths[i] + 1)
				queue.append(queue[i].children[val])
				values.append(val)
			f_acc.write(str(depths[i]) + ',' + str(self.test_accuracy(test_data)) + '\n')
			i += 1
		# following lines for testing
		# we can add more checks
		tot = 0
		for i in range(len(queue) - 1, 0, -1):
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
			print("pruning_run =", run + 1)
			mx_acc = 0
			mx_acc_node = self.root
			while i < len(queue):
				ps_ch = dict(queue[i].children)
				queue[i].children = {}
				ps_acc = self.test_accuracy(validation)
				if ps_acc > mx_acc:
					mx_acc = ps_acc
					mx_acc_node = queue[i]
				queue[i].children = ps_ch
				for val in (queue[i].children):
					queue.append(queue[i].children[val])
				i += 1
			run += 1
			if mx_acc > org_acc:
				mx_acc_node.children = {}
			else:
				break
		print("final_validation_acc =",org_acc)

org_data = data()
org_data.readByFile('data/train.csv')
org_data.FillMissingVal()

train_data, test_com_data = org_data.split(0.7)
valid_data, test_data = test_com_data.split(0.5)
# train_data = org_data
# test_data = org_data

model = DecisionTree('is_patient')
model.train(train_data, 'information_gain', test_data)
print("starting_test_acc =", model.test_accuracy(test_data))
model.prune(valid_data)
print("test_acc =", model.test_accuracy(test_data))

model.train(train_data, 'gini_gain', test_data)
print("starting_test_acc =", model.test_accuracy(test_data))
model.prune(valid_data)
print("test_acc =", model.test_accuracy(test_data))
