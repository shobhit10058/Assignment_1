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
				ps_e = inp.readline(',')

				if len(ps_e) == 0:
					break
				
				for col in range(ps_e):
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

	def entropy(self):
		class_count = self.getclassCount()
		entropy_value = 0

		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[-1])
			entropy_value -= (class_count[vals] * log2(class_count[vals]))
		
		return entropy_value

	def GinnnyIndex(self):
		class_count = self.getclassCount()
		giny_indx = 1
		
		for vals in class_count:
			class_count[vals] /= len(self.examples.values_of_attributes[-1])
			giny_indx -= (class_count[vals] ** 2)
			
		return giny_indx

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

	def BestGinyGain(self):
		# check if pure node
		pass

	def BestInformationGain(self):
		# check if pure node
		pass


class DecisionTree:

	def __init__(self) -> None:
		self.root = node()

	def train(examples: data) -> None:
		pass
