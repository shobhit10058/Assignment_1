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
