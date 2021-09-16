from math import log2
from Data import data

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

	def predictTest(self, test: data):
		predictions = []
		for ex_ind in range(len(test.values_of_attributes[self.target_attr])):
			test_inst = {}
			for attr in test.attributes:
				test_inst[attr] = (test.values_of_attributes[attr][ex_ind])
			predictions.append(self.predictInstance(test_inst))
		return predictions

	def test_accuracy(self, test: data) -> float:
		predictions = self.predictTest(test)
		gt_corr = 0
		tot = len(predictions)
		for i in range(tot):
			gt_corr += (predictions[i] == test.values_of_attributes[self.target_attr][i])
		return gt_corr / tot
	
	def train(self, examples: data, heuristic: str, test: data, track_acc: bool) -> None:
		# train in bfs format
		self.root.giveExamples(examples)
		queue = [self.root]
		depths = [0]
		i = 0
		if track_acc:
			f_acc_d = open(heuristic + "_model_acc_chang_with_depth" + ".csv", 'w')
			f_acc_n = open(heuristic + "_model_acc_chang_with_number_of_nodes" + ".csv", 'w')
			f_acc_d.write("depth,accuracy\n")
			f_acc_n.write("number of nodes,accuracy\n")
		while i < len(queue):
			ps_acc = self.test_accuracy(test)
			if track_acc:
				f_acc_d.write(str(depths[i]) + ',' + str(ps_acc) + '\n')
				f_acc_n.write(str(len(queue)) + ',' + str(ps_acc) + '\n')
			queue[i].splitByHeuristic(heuristic)
			for child in (queue[i].children):
				depths.append(depths[i] + 1)
				queue.append(child)
			i += 1

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
			if queue[i].split_attr != "none":
				f.write("[ " + str(queue[i].split_attr) + " > " + str(queue[i].split_attr_thrs) + " ]\t")
			else:
				f.write("[ target value = " + (str)(int(queue[i].getClassCount()[1])) + " ]\t")
			for child in (queue[i].children):
				depths.append(depths[i] + 1)
				queue.append(child)
			i += 1