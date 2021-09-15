from math import log2
import random

class data:
	
	def __init__(self) -> None:
		self.attributes = []
		self.values_of_attributes = []	
	
	def readByFile(self, input_file: str) -> None:
		
		with open(input_file) as inp:
			self.attributes = inp.readline()[:-1].split(',')
			self.values_of_attributes = []

			for attr in self.attributes:
				self.values_of_attributes.append([])
			
			while(1):
				read_str = inp.readline()[:-1]
				if len(read_str) == 0:
					break
				ps_e = read_str.split(',')
				for col in range(len(ps_e)):
					self.values_of_attributes[col].append(ps_e[col])

	def initAttr(self, attributes: list):
		self.attributes = attributes
		self.values_of_attributes = [[] for _ in range(len(self.attributes))]

	def addExample(self, values):
		for attr_indx in range(len(values)):
			self.values_of_attributes[attr_indx].append(values[attr_indx])

	def FillMissingVal(self) -> None:
		for attr_ind in range(len(self.attributes)):
			co_vals = {}
			ms_val = ""
			ms_co  = 0
			
			for val_ind in range(len(self.values_of_attributes[attr_ind])):
				ps_v = self.values_of_attributes[attr_ind][val_ind]
				if len(ps_v) == 0:
					continue
				if not ps_v in co_vals:
					co_vals[ps_v] = 0
				co_vals[ps_v] += 1
				if ms_co <= co_vals[ps_v]:
					ms_co = co_vals[ps_v]
					ms_val = ps_v
			
			for val_ind in range(len(self.values_of_attributes[attr_ind])):
				ps_v = self.values_of_attributes[attr_ind][val_ind]
				if len(ps_v) == 0:
					self.values_of_attributes[attr_ind][val_ind] = ms_val
					continue

	def split(self, test_frac):
		data_size = len(self.values_of_attributes[0])
		test_size = int(test_frac * data_size)
		indices = [_ for _ in range(data_size)]
		chosen_test_ind = set(random.sample(indices, test_size))
		train_data = data()
		test_data = data()
		train_data.initAttr(self.attributes)
		test_data.initAttr(self.attributes)
		for ex_ind in range(data_size):
			example = []
			for attr_ind in range(len(self.attributes)):
				example.append(self.values_of_attributes[attr_ind][ex_ind])
			if ex_ind in chosen_test_ind:
				test_data.addExample(example)
			else:
				train_data.addExample(example)
			
		return (train_data, test_data)

class node:

	def __init__(self) -> None:
		self.examples = data()
		self.split_attr_indx = -1
		# list of nodes
		self.children = {}

	def giveAttributes(self, attributes: list) -> None:
		self.examples.initAttr(attributes)
	
	def addExample(self, example: list) -> None:
		self.examples.addExample(example)

	def giveExamples(self, examples: data) -> None:
		self.examples = examples
	
	def getClassCount(self):
		class_count = {}

		for entry in self.examples.values_of_attributes[-1]:
			if not entry in class_count:
				class_count[entry] = 0
			class_count[entry] += 1

		return class_count

	def splitNode(self, attr_indx):
		map_val_to_children = {}

		ps_attrs = self.examples.attributes[:attr_indx]
		ps_attrs.extend(self.examples.attributes[attr_indx + 1:])
		
		for ps_ex_indx in range(len(self.examples.values_of_attributes[attr_indx])):
			ps_val = self.examples.values_of_attributes[attr_indx][ps_ex_indx]
			
			if not ps_val in map_val_to_children:
				map_val_to_children[ps_val] = node()
				map_val_to_children[ps_val].giveAttributes(ps_attrs)
			
			ps_example = []
			
			for ps_attr_indx in range(len(self.examples.attributes)):
				if ps_attr_indx != attr_indx:
					ps_example.append(self.examples.values_of_attributes[ps_attr_indx][ps_ex_indx])

			map_val_to_children[ps_val].addExample(ps_example)

		return map_val_to_children
	
	def entropy(self):
		class_count = self.getClassCount()
		entropy_value = 0

		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[-1])
			entropy_value -= (class_count[vals] * log2(class_count[vals]))
		
		return entropy_value

	def ginnnyIndex(self):
		class_count = self.getClassCount()
		giny_indx = 1
		
		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[-1])
			giny_indx -= (class_count[vals] ** 2)
			
		return giny_indx

	def informationGain(self, attr_indx):
		entropy_change = self.entropy()
		cont_child_nodes = self.splitNode(attr_indx)
		
		for val in cont_child_nodes:
			prob = len(cont_child_nodes[val].examples.values_of_attributes[-1]) / len(self.examples.values_of_attributes[-1])
			entropy_change -= (prob * cont_child_nodes[val].entropy())

		return entropy_change

	def giniGain(self, attr_indx):
		gini_change = self.ginnnyIndex()
		cont_child_nodes = self.splitNode(attr_indx)
		
		for val in cont_child_nodes:
			prob = len(cont_child_nodes[val].examples.values_of_attributes[-1]) / len(self.examples.values_of_attributes[-1])
			gini_change -= (prob * cont_child_nodes[val].ginnnyIndex())

		return gini_change

	# assigns the children by splitting with best gain attribute	
	def splitByHeuristic(self, heuristic) -> None:
		# check if pure node
		if(self.entropy() == 0):
			return
		best_gain = -1
		best_attr_indx = -1

		for cont_attr_index in range(len(self.examples.attributes) - 1):
			ps_gain = 0
			if(heuristic == 'information_gain'):
				ps_gain = self.informationGain(cont_attr_index)
			else:
				ps_gain = self.giniGain(cont_attr_index)
			
			if ps_gain > best_gain:
				best_gain = ps_gain
				best_attr_indx = cont_attr_index
			
		if(best_attr_indx == -1):
			return	
		
		self.children = self.splitNode(best_attr_indx)
		self.split_attr_indx = best_attr_indx

class DecisionTree:

	def __init__(self) -> None:
		self.root = node()

	def recursion_train(self, root, heuristic: str):
		root.splitByHeuristic(heuristic)
		for ch_val in root.children:
			self.recursion_train(root.children[ch_val], heuristic)
	
	def predictInstance(self, example: list):
		ps_node = self.root
		while(len(ps_node.children) > 0):
			ps_val = example[ps_node.split_attr_indx]
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
		for ex_ind in range(len(test.values_of_attributes[0])):
			test_inst = []
			for attr_ind in range(len(test.attributes)):
				test_inst.append(test.values_of_attributes[attr_ind][ex_ind])
			predictions.append(self.predictInstance(test_inst))
		return predictions

	# implement the following accuracy test
	def test_accuracy(self, test: data) -> float:
		predictions = self.predictTest(test)
		gt_corr = 0
		tot = len(predictions)
		for i in range(tot):
			gt_corr += (predictions[i] == test.values_of_attributes[-1][i])
		return gt_corr / tot
	
	# add the procedures to add accuracy check at each depth
	def train(self, examples: data, heuristic: str, test: data) -> None:
		# train in bfs format
		self.root.giveExamples(examples)
		queue = []
		depths = []
		queue.append(self.root)
		depths.append(0)
		i = 0
		f_acc = open(heuristic + "_model_acc_chang" + ".csv", 'w')
		f_acc.write("depth,accuracy\n")
		while i < len(queue):
			queue[i].splitByHeuristic(heuristic)
			for val in (queue[i].children):
				depths.append(depths[i] + 1)
				queue.append(queue[i].children[val])
			f_acc.write(str(depths[i]) + ',' + str(self.test_accuracy(test_data)) + '\n')
			i += 1
		# following lines for testing
		# we can add more checks
		tot = 0
		for i in range(len(queue) - 1, 0, -1):
			if len(queue[i].children) == 0:
				tot += len(queue[i].examples.values_of_attributes[0])
		if tot != len(examples.values_of_attributes[0]):
			print("Something fishy")
		else:
			print("Tree is proper")

		# uncomment following for recursive training
		# self.recursion_train(self.root, heuristic)

org_data = data()
org_data.readByFile('data/train.csv')
org_data.FillMissingVal()

train_data, test_data = org_data.split(0.2)

model = DecisionTree()
model.train(train_data, 'information_gain', test_data)
print(model.test_accuracy(test_data))

model.train(train_data, 'gini_gain', test_data)
print(model.test_accuracy(test_data))
