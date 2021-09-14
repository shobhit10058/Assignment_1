from math import log2

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
			
			line = 0
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
		pass

	def split(self, test_frac):
		pass

class node:

	def __init__(self) -> None:
		self.examples = data()
		# list of nodes
		self.children = []

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
		child_nodes = {}

		ps_attrs = self.examples.attributes[:attr_indx]
		ps_attrs.extend(self.examples.attributes[attr_indx + 1:])
		
		for ps_ex_indx in range(len(self.examples.values_of_attributes[attr_indx])):
			ps_val = self.examples.values_of_attributes[attr_indx][ps_ex_indx]
			
			if not ps_val in child_nodes:
				child_nodes[ps_val] = node()
				child_nodes[ps_val].giveAttributes(ps_attrs)
			
			ps_example = []
			
			for ps_attr_indx in range(len(self.examples.attributes)):
				if ps_attr_indx != attr_indx:
					ps_example.append(self.examples.values_of_attributes[ps_attr_indx][ps_ex_indx])

			child_nodes[ps_val].addExample(ps_example)

		cont_child_nodes = []

		for ch_attr_val in child_nodes:
			cont_child_nodes.append(child_nodes[ch_attr_val])
		
		return cont_child_nodes
	
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
		
		for cont_child in cont_child_nodes:
			prob = len(cont_child.examples.values_of_attributes[-1]) / len(self.examples.values_of_attributes[-1])
			entropy_change -= (prob * cont_child.entropy())

		return entropy_change

	def giniGain(self, attr_indx):
		gini_change = self.ginnnyIndex()
		cont_child_nodes = self.splitNode(attr_indx)
		
		for cont_child in cont_child_nodes:
			prob = len(cont_child.examples.values_of_attributes[-1]) / len(self.examples.values_of_attributes[-1])
			gini_change -= prob * (cont_child.ginnnyIndex())

		return gini_change

	# assigns the children by splitting with best gain attribute	
	def splitByHeuristic(self, heuristic) -> None:
		# check if pure node
		if(self.entropy() == 0):
			return
		best_gain = 0
		best_attr_indx = -1

		for cont_attr_index in range(len(self.examples.attributes) - 1):
			ps_gain = 0
			if(heuristic == "information_gain"):
				ps_gain = self.informationGain(cont_attr_index)
			else:
				ps_gain = self.giniGain(cont_attr_index)
			
			if ps_gain > best_gain:
				best_gain = ps_gain
				best_attr_indx = cont_attr_index
			
		if(best_attr_indx == -1):
			return	
		
		self.children = self.splitNode(best_attr_indx)

class DecisionTree:

	def __init__(self) -> None:
		self.root = node()

	def recursion_train(self, root, heuristic: str):
		root.splitByHeuristic(heuristic)
		for ch_node in root.children:
			self.recursion_train(ch_node, heuristic)
	
	# use the value of target attribute stored in -1 index in values of attributes
	# in leaf node at which test data arrives
	def predict(self, test: data):
		pass

	# implement the following accuracy test
	def test_accuracy(self, test: data) -> None:
		pass

	# add the procedures to add accuracy check at each depth
	def train(self, examples: data, heuristic: str) -> None:
		# train in bfs format
		self.root.giveExamples(examples)
		queue = []
		depths = []
		queue.append(self.root)
		depths.append(0)
		i = 0
		while i < len(queue):
			queue[i].splitByHeuristic(heuristic)
			queue.extend(queue[i].children)
			for _ in range(len(queue[i].children)):
				depths.append(depths[i] + 1)
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

train_data = data()
train_data.readByFile('data/train.csv')

model = DecisionTree()
model.train(train_data, 'information_gain')
model.train(train_data, 'gini_gain')