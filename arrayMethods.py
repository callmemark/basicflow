from arrayObj import Array
class ArrayMethods():
	def __init__(self):
		pass


	def matrixMultiply(self, multiplier_arr, arr_to_mult):
		"""
			matrix multiplt two array
			
			Argumemts:
			multiplier_arr	(List / Array)	
			arr_to_mult		(List / Array)

			Return Array
		"""
		
		output_array = []

		for multiplier in multiplier_arr:
			row_arr = []
			for value in arr_to_mult:
				product = multiplier * value
				row_arr.append(product)

			output_array.append(row_arr)

		return output_array



	def matrixVectorMultiply(self, multiplicand_arr, multiplier_arr):
		"""
			Calculate dot product of two array
			
			Argumemts:
			multiplicand_arr	(2d matrix array) 
			multiplier_arr		(1d vector)

			Return Array
		"""
		output_array = []

		for i in range(len(multiplicand_arr)):
			arr = multiplicand_arr[i]
			this_arr = []
			for j in arr:
				this_arr.append(j * multiplier_arr[i])
			output_array.append(this_arr)

		return output_array
	
		




	def getMatrixSumOfRow(self, _2d_matrix):
		"""
			caculate the sum of rows of a 2d matrix array
			
			Argumemts:
			_2d_matrix	(2d matrix Array)

			Return Matrix Array
		"""
		final_arr = []

		for selected_row in _2d_matrix:
			sum_of_current_iter = 0
			for value in selected_row:
				sum_of_current_iter += value

			final_arr.append([sum_of_current_iter])

		return final_arr



	def vectorMultiply(self, vector_array_01, vector_array_02):
		"""
			Multiply two 1d vector
			
			Argumemts:
			vector_array_01	(1d vector Array)
			vector_array_02 (1d vector Array)

			Return vector array
		"""
		array_01_lenght = len(vector_array_01)
		array_02_lenght = len(vector_array_02)

		if array_01_lenght != array_02_lenght:
			raise Exception("Error: Array are not equal where size is ", array_01_lenght, " != ", array_02_lenght)

		output_array = []
		loop_n = int((array_01_lenght + array_02_lenght) / 2)

		for index in range(loop_n):
			output_array.append(vector_array_01[index] * vector_array_02[index])

		return output_array



	def matrixAddition(self, matrix_01, matrix_02):
		"""
			add two matrix
			
			Argumemts:
			matrix_01	(2d matrix Array)
			matrix_02	(2d matrix Array)

			Return 2d Matrix Array
		"""

		if len(matrix_01) != len(matrix_02) or len(matrix_01[0]) != len(matrix_02[0]):
			raise ValueError("Arrays must have the same shape")
		
		result = [[0 for _ in range(len(matrix_01[0]))] for _ in range(len(matrix_01))]
		for i in range(len(matrix_01)):
			for j in range(len(matrix_01[0])):
				result[i][j] = matrix_01[i][j] + matrix_02[i][j]

		return result



	def matrixSubtract(self, matrix_minuend, matrix_subtrahend):
		"""
			subtract two matrix
			
			Argumemts:
			matrix_minuend	(2d matrix Array)
			matrix_subtrahend	(2d matrix Array)

			Return 2d Matrix Array
		"""
		if len(matrix_minuend) != len(matrix_subtrahend) or len(matrix_minuend[0]) != len(matrix_subtrahend[0]):
			raise ValueError("Arrays must have the same shape")

		result = [[0 for _ in range(len(matrix_minuend[0]))] for _ in range(len(matrix_minuend))]

		for i in range(len(matrix_minuend)):
			for j in range(len(matrix_minuend[0])):
				result[i][j] = matrix_minuend[i][j] - matrix_subtrahend[i][j]

		return result


	def vectorSubtract(self, minuend_vector_array, subtrahend_vector_array):
		"""
			subtract two matrix
			
			Argumemts:
			minuend_vector_array	(1d vector Array)
			subtrahend_vector_array	(1d vector Array)

			Return 1d vector Array
		"""
		subtracted_arr = []
		minuend_arr_size = len(minuend_vector_array)
		subtrahend_arr_size = len(subtrahend_vector_array)

		if minuend_arr_size != subtrahend_arr_size:
			raise Exception(str("Error on function 'vectorSubtract'. Arrays are not equal lenght with sizes " + str(minuend_arr_size) + " and " + str(subtrahend_arr_size)))

		index_count = int((minuend_arr_size + subtrahend_arr_size) / 2)
		for index in range(index_count):
			subtracted_arr.append(minuend_vector_array[index] - subtrahend_vector_array[index])

		return subtracted_arr


	def flatten(self, matrix_arr):
		"""
			transform a matrix array into a 1d vector array
			
			Argumemts:
			matrix_arr	(nd vector Array)

			Return 1d vector Array
		"""
		flattened = []

		if len(self.getShape(matrix_arr)) == 1:
			return matrix_arr

		for element in matrix_arr:
			if isinstance(element, list):
				flattened.extend(self.flatten(element))
			else:
				flattened.append(element)
		
		return flattened


	def getShape(self, array_arg):
		"""
			get the shape of vector / matrix array
			
			Argumemts:
			array_arg	(nd Array)

			Return 1d vector Array
		"""
		shape = []

		while isinstance(array_arg, list):
			shape.append(len(array_arg))
			array_arg = array_arg[0]

		return shape


	def transpose(self, arr_arg):
		""""
			Transpose the given 2d matrix array swapping its elemtns positions

			Arguements			:	arr_arg
			Returns(Array) 		:	matrix array
		"""

		if len(self.getShape(arr_arg)) <= 1:
			return arr_arg

		shape = self.getShape(arr_arg)
		transposed_list = [[None]*shape[0] for _ in range(shape[1])]

		for i in range(shape[0]):
			for j in range(shape[1]):
				transposed_list[j][i] = arr_arg[i][j]

		return transposed_list


	def vectorSquare(self, vector_array):
		""""
			Transpose the given 2d matrix array swapping its elemtns positions

			Arguements			:	arr_arg
			Returns(Array) 		:	matrix array
		"""
		squared_arr = []
		arr_shape = self.getShape(vector_array)
		if len(arr_shape) != 1:
			raise Exception("Error in function vectorSquare, Array should be one dimesion but have " + str(arr_shape))

		for value in vector_array:
			squared_arr.append(value ** 2)

		return squared_arr


	def vectorScalarMultiply(self, vector_array, multiplier_num):
		""""
			apply scalar multiplication to the given array using the given multiplier

			Arguements:
			vector_array (1d vector array)		:	1d vector array
			multiplier_num (float) 				:	float

			Returns: 1d vector array
		"""
		resulting_arr = []

		for value in vector_array:
			resulting_arr.append(value * multiplier_num)

		return resulting_arr


	def matixScalaMultiply(self, matrix, scalar_multiplier):
		output_array = []

		for row in matrix:
			col_val_sum = []
			for col_val in row:
				col_val_sum.append(col_val * scalar_multiplier)

			output_array.append(col_val_sum)

		return Array(output_array)