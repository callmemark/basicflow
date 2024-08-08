from random import uniform
from arrayObj import Array

class WeightInitializationMethods():
	def __init__(self):
		"""
			Methods to intializ random value generated using different mathematical functions

			Arguments: Takes 0 arguments
		"""
		#super().__init__()
		pass



	def radomInitializer(self, min_f = 0, max_f = 1.0):
		"""
			Generate random number in range of given paramenter using basic calculation technique

			Arguments: 
			min_f (float) 	:	The minimum value limit
			max_f (float)	:	The maximum value limit

			Returns:float
		"""
		rwg = 2 * uniform(min_f, max_f) - 1

		return rwg


	def NormalizedXavierWeightInitializer(self, col_size, n_of_preceding_nodes, n_of_proceding_node):
		"""
			Generate random number using xavier weight intializer 

			Arguments: 
			col_size (float) 				:	the number of elements or weights to be generated since this will be a 1d array
			n_of_preceding_nodes (Array)	:	The number of neurons where outputs will come from
			n_of_proceding_node (Array)		:	The number of neurons that will accepts the outputs frrom the preceding neuro

			Returns:Array
		"""
		n = n_of_preceding_nodes
		m = n_of_proceding_node

		sum_of_node_count = n + m

		lower_range, upper_range = -(sqrt(6.0) / sqrt(sum_of_node_count)), (sqrt(6.0) / sqrt(sum_of_node_count))
		rand_num = Array([uniform(0, 1) for i in range(col_size)])
		scaled = rand_num.add(lower_range).multiply((upper_range - lower_range))

		return Array(scaled)








class WeightInitializer(WeightInitializationMethods):
	def __init__(self):
		"""
			This class contains different methods to generate weights tot thee neural network

			Arguments: takes 0 arguments
		"""
		super().__init__()


	def intializeWeight(self, dim, min_f = -1.0, max_f = 1.0):
		"""
			This method generate weights using simple random number calculations

			Arguments: 
			dim (lsit)		: 	A two lenght list contains the row and columnn [row, col] or shape of the generated weight
			min_f (float) 	:	The minimum value limit
			max_f (float)	:	The maximum value limit
			

			Returns:Array
		"""
		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for i in range(row):
			col_arr = []
			for j in range(col):
				col_arr.append(self.radomInitializer(min_f, max_f))

			final_weight_arr.append(col_arr)

		return Array(final_weight_arr)


	def initNormalizedXavierWeight(self, dim, n_of_preceding_nodes, n_of_proceding_node):
		"""
			This method generate weights using xavier weight initialization method

			Arguments: 
			dim (list)		: 	A two lenght list contains the row and columnn [row, col] or shape of the generated weight
			n_of_preceding_nodes (Array)	:	The number of neurons where outputs will come from
			n_of_proceding_node (Array)		:	The number of neurons that will accepts the outputs frrom the preceding neuro

			Returns:Array
		"""

		final_weight_arr = []
		row = dim[0]
		col = dim[1]

		for row_count in range(row):
			data = self.NormalizedXavierWeightInitializer(col, n_of_preceding_nodes, n_of_proceding_node)
			final_weight_arr.append(data)

		return Array(final_weight_arr)