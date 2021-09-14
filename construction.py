from math import log2

class data:
	
	def __init__(self) -> None:
		self.attributes = []
		self.values_of_attributes = []	
	
	def readByFile(self, input_file: str) -> None:
		
		with open(input_file) as inp:
			self.attributes = inp.readline().split(',')
			self.values_of_attributes = []

			for attr in self.attributes:
				self.values_of_attributes.append([])
				self.attributes.append(attr)
			
			while(1):
				ps_e = inp.readline().split(',')

				if len(ps_e) == 0:
					break
				
				for col in range(len(ps_e)):
					self.values_of_attributes[col].append(ps_e[col])

	def initAttr(self, attributes: list):
		self.attributes = attributes

	def addExample(self, values):
		for attr_indx in range(len(values)):
			self.values_of_attributes.append(values[attr_indx])

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

			child_nodes[ps_val].giveAttributes(ps_attr_indx)
			ps_example = []
			
			for ps_attr_indx in range(len(self.examples.attributes)):
				if ps_attr_indx != attr_indx:
					ps_example.append(self.examples.values_of_attributes[ps_attr_indx][ps_ex_indx])

			child_nodes[ps_val].addExample(ps_example)

		return child_nodes
	
	def entropy(self):
		class_count = self.getclassCount()
		entropy_value = 0

		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[-1])
			entropy_value -= (class_count[vals] * log2(class_count[vals]))
		
		return entropy_value

	def ginnnyIndex(self):
		class_count = self.getclassCount()
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
			entropy_change -= prob * cont_child.entropy()

		return entropy_change

	def giniGain(self, attr_indx):
		gini_change = self.ginnnyIndex()
		cont_child_nodes = self.splitNode(attr_indx)
		
		for cont_child in cont_child_nodes:
			prob = len(cont_child.examples.values_of_attributes[-1]) / len(self.examples.values_of_attributes[-1])
			gini_change -= prob * cont_child.ginnnyIndex()

		return gini_change
	
	# assigns the children by splitting with best giny gain attribute
	def splitByBestGinyGain(self) -> None:
		# check if pure node
		if(self.ginnnyIndex() == 0):
			return
		best_gain = 0
		best_attr_indx = 0

		for cont_attr_index in range(len(self.examples.attributes) - 1):
			ps_gain = self.giniGain(cont_attr_index)

			if ps_gain > best_gain:
				best_gain = ps_gain
				best_attr_indx = cont_attr_index
			
		new_child_nodes = self.splitNode(best_attr_indx)
		self.children = []

		for ch_attr_val in new_child_nodes:
			self.children.append(new_child_nodes[ch_attr_val])

	# assigns the children by splitting with best information gain attribute	
	def splitByBestInformationGain(self) -> None:
		# check if pure node
		if(self.entropy() == 0):
			return
		best_gain = 0
		best_attr_indx = 0

		for cont_attr_index in range(len(self.examples.attributes) - 1):
			ps_gain = self.informationGain(cont_attr_index)

			if ps_gain > best_gain:
				best_gain = ps_gain
				best_attr_indx = cont_attr_index
			
		new_child_nodes = self.splitNode(best_attr_indx)
		self.children = []
		
		for ch_attr_val in new_child_nodes:
			self.children.append(new_child_nodes[ch_attr_val])

class DecisionTree:

	def __init__(self) -> None:
		self.root = node()

	def recursion_train(self, root, heuristic: str):
		if(heuristic == "information_gain"):
			root.splitByBestInformationGain()
		else:
			root.splitByBestGinyGain()
		for ch_node in root.children:
			self.recursion_train(ch_node, heuristic)

	def train(self, examples: data, heuristic: str) -> None:
		# train in bfs format
		self.root.giveExamples(examples)
		self.recursion_train(self.root, heuristic)

train_data = data()
train_data.readByFile('data/train.csv')

model = DecisionTree()
model.train(train_data, 'information_gain')