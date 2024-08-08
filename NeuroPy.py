
"""

Author: Mark John Velmonte
Date: February 2, 2023

Description: Contains class to create simple but expandable neural network from scratch.


"""



from math import log, sqrt
from arrayMethods import ArrayMethods
from arrayObj import Array
from weightInitializer import WeightInitializer

try:
	from tqdm import tqdm
except:
	raise Exception("Missing required library tqdm: please install tqdm via 'pip install tqdm'")












class ActivationFunction():
	def __init__(self):
		"""
			This class contains different methods that calculate different deep learning functions
			Arguments: takes 0 arguments
		"""
		self.E = 2.71


	def sigmoidFunction(self, x):
		"""
			This method perform a sigmoid function calculation

			Arguments: 
			x(float) 	: The value where sigmoid function will be applied
			
			Returns: float
		"""
		x = x
		result = 1 / (1 + self.E ** -x)

		return result


	def argMax(self, ouput_vector):
		"""
			This method search for the maximum value and create a new list where only the maximum value will have a value of 1

			Arguments: 
			weight_matrix (Matrix) 	: The array that will be transformed into a new array
			
			Returns: Array
		"""
		output_array = []

		max_value_index = ouput_vector.index(max(ouput_vector))

		for index in range(len(ouput_vector)):
			if index == max_value_index:
				output_array.append(1)
			elif index != max_value_index:
				output_array.append(0)

		return Array(output_array)








class ForwardPropagation(ActivationFunction):
	def __init__(self):
		"""
			This class contains different methods for neural network forward propagation

			Arguments: takes 0 arguments
		"""
		super().__init__()


	def fowardPropagation(self, input_value, weight_value, bias_weight, activation_function):
		"""
			Creates a nueral network layer

			Arguments: 
			input_value (Array) 	: 	testing inputs 
			weight_value (Array)	:	The corresponding weight to this layer
			bias_weight (Array)		:	The weight of the bias for this layer
			
			Returns (Array) : The ouput of the this layer
		"""
		weighted_sum = self.getWeightedSum(input_value, weight_value)
		biased_weighted_sum = Array(weighted_sum).addArray(ArrayMethods().flatten(bias_weight))

		if activation_function == "sigmoid":
			result = self.sigmoidNeuronActivation(biased_weighted_sum)

		return Array(result)



	def sigmoidNeuronActivation(self, input_vector):
		"""
			Handles neuron activation

			Arguments: 
			input_vector (Matrix) 	: 	Expects the matrix of weighted sum 
			
			Returns (Matrix)
		"""

		result = []
		for input_val in input_vector:
			result.append(self.sigmoidFunction(input_val))

		return result


	def getWeightedSum(self, input_arr, weight_arr):
		"""
			Caculate weighted sum of the incoming input

			Arguments: 
			input_arr (Array) 	: 	Inputs eigther from a layer ouput or the main testing data
			weight_arr (Array)	: 	The generated weight
			
			Returns (Array) : Weighted sum 
		"""

		weighted_sum_arr = []
		for row in weight_arr:
			sum_of_product = 0
			for index in range(len(row)):
				sum_of_product += (row[index] * input_arr[index])

			weighted_sum_arr.append(sum_of_product)

		return weighted_sum_arr


	def applyBias(self, bias_weight_arr, weighted_sum_arr):
		"""
			apply the bias to the incoming inputs to the recieving neurons layer

			Arguments: 
			bias_weight_arr (Array) 		: 	weights of the bias to be added to the incoming inputs
			weighted_sum_arr (Array)		: 	The generated weight
			
			Returns (Array) : biased inputs
		"""
		return Array(weighted_sum_arr).add(bias_weight_arr)















class WeightUpdates():
	def __init__(self):
		super().__init__()


	def sigmoidWeightCalculation(self, succeeding_layer_neuron_strenght, preceding_neuron_output, initial_weight_matrix):
		""" 
			Calculate and return an array of floats that is intended to use for calibrating the weights of the 
			Neural network
			
			Arguments:
			succeeding_layer_neuron_strenght (List / Array)	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			preceding_neuron_output (List / Array)			:	The layer of neurons that is first to recieve data relative to forward propagation direction
			initial_weight_matrix (list / Matrix)			:	The initial weight without the adjustments

			Returns: Array

			formula:
			weight_ajustments = -learning_rate * [matrixMultiply(succeeding_layer_neuron_strenght, preceding_neuron_output)]
		"""

		neighbor_neuron_dprod = self.matrixMultiply(succeeding_layer_neuron_strenght, preceding_neuron_output)

		weight_adjustment_matrix = []
		for selected_row in neighbor_neuron_dprod:
			result_row = []

			for col_val in selected_row:
				product = self.learning_rate * col_val
				result_row.append(product)

			weight_adjustment_matrix.append(result_row)


		weight_update_matrix = self.applyWeightAdjustment(
						initial_weight = initial_weight_matrix, 
						weight_adjustment = weight_adjustment_matrix, 
						operation = "+"
					)

		return weight_adjustment_matrix



	def sigmoidL2RegularizationWeightUpdate(self, learning_rate, delta_n, prev_layer_output, l2_lambda, intial_weight):
		"""
			Apply the l2 regularization when calculating the weights of the layers

			Aguemtns:
			learning_rate (float)			: The models learning rate
			delta_n (Vector)				: The strenght of the hidden layer recieving from the weihts being update
			prev_layer_output (Matrix)		: The output of the sactivation function of the previos layer
			l2_lambda (float) 				: L2 penalty
			intial_weight (matrix)			: The initial weight

			return matrix with values representing the amount to change the weight
 
		"""

		# delta_w = alpha * (delta * X.T) + lambda * W
		delta_w = self.matrixAddition(
								Array(delta_n).vectorMultiply(Array(prev_layer_output).multiply(learning_rate)), 
								self.matixScalaMultiply(intial_weight, l2_lambda)
								)
		

		# W = W - (alpha * dW + lambda * W)
		weight_matrix_update = self.matrixSubtract(
								intial_weight,
								self.matrixAddition(
									self.matixScalaMultiply(delta_w, learning_rate), 
									self.matixScalaMultiply(intial_weight, l2_lambda)
									)
								) 

		return Array(weight_matrix_update)







	def applyWeightAdjustment(self, initial_weight, weight_adjustment, operation = "+"):
		"""
			Apply the adjustments of the weights to the initial weight to update its value by getting the sum of the two array

			Arguments:
			initial_weight (List / Array)			:	The weights value that is used in forward propagation
			weight_adjustment  (List / Array)		:	The value used to add to the initial weight

			Returns: Array
		"""
		if operation == "+":
			returned_value = self.matrixAddition(initial_weight, weight_adjustment)
		elif operation == "-":
			returned_value = self.matrixSubtract(initial_weight, weight_adjustment)

		return Array(returned_value)










class DeltaCalculationMethods():
	def __init__(self):
		super().__init__()


	def sigmoidDeltaCalculation(self, preceding_neuron_output_vector, weight_matrix, proceding_neuron_output_matrix):
		"""
			Calculate the delta of the a layer using sigmoid derivative

			Arguments: 
				preceding_neuron_output_vector (vector) (ith_layer - 1) 
				weight_matrix (matrix)
				proceding_neuron_output_matrix (matrix)  (ith_layer + 1) 

			Return (Vertor) Returns the calculated delta of the layer
		"""
		transposed_weight = self.transpose(weight_matrix)

		subtracted_arr = []
		for neuron_val in preceding_neuron_output_vector:
			subtracted_arr.append(1 - neuron_val)

		product_arr = []
		for index in range(len(preceding_neuron_output_vector)):
			product_arr.append(preceding_neuron_output_vector[index] * subtracted_arr[index])

		dot_product_arr = self.matrixVectorMultiply(transposed_weight, proceding_neuron_output_matrix)
		sum_of_rows_arr = self.getMatrixSumOfRow(dot_product_arr)
		neuron_strenghts = self.vectorMultiply(self.flatten(sum_of_rows_arr), product_arr)

		return Array(neuron_strenghts)









class BackPropagation(ArrayMethods, Array, WeightUpdates, DeltaCalculationMethods):
	def __init__(self, learning_rate = -0.01):
		"""
			This class handles the backpropagation acalculation methods
		"""

		super().__init__()
		self.learning_rate = learning_rate


	def getCrossEntropyLoss(self, predicted_ouputs_vector, actual_label_vector):
		"""
			This method is made to calculate the coss entropy loss for the final layer

			Arguments:
			predicted_ouputs_vector (Vector) (p)		:	Networks final layer or prediction
			actual_label_vector (Vector) (y)	:	The actual label for the given problem

			Return (Vector) calculated loss
			Equation : -y * log(p) - (1 - y) * log(1-p)
		"""
		try:
			output_vector = []

			for value_index in range(len(predicted_ouputs_vector)):
				y = actual_label_vector[value_index]
				p = predicted_ouputs_vector[value_index]

				#print("p = ", p, " y = ", y)
				output = -y * log(p+1e-9) - (1 - y) * log(1 - p+1e-9)

				output_vector.append(output)

			return output_vector
		except ValueError:
			err_msg = "Math dommain erro: where p = " + str(p)
			raise Exception(err_msg)



	# rename from getFLayerNeuronStrenght to getFinalLayerDelta
	def getFinalLayerDelta(self, predicted_ouputs_vector, actual_label_vector):
		"""
			Calculate the final layer neuron strenghts

			Arguments:
			predicted_ouputs_vector (List / Array)				:	Final output that is calculated by sigmoid function
			actual_label_vector (List / Array)	:	The final ouput that is produced by argmax function

			Returns: Array

		"""
		#returned_value = Array(self.vectorSubtract(predicted_ouputs_vector, actual_label_vector)).squared()
		returned_value = self.vectorSubtract(predicted_ouputs_vector, actual_label_vector)
		return Array(returned_value)



	

	# rename from calculateWeightAdjustment to updateLayerWeight
	def updateLayerWeight(self, succeeding_layer_neuron_strenght, preceding_neuron_output, initial_weight_matrix, activation_function = "sigmoid"):
		
		"""
			Update weight matrix

		"""

		if activation_function == "sigmoid":
			weight_update_matrix = self.sigmoidWeightCalculation(
										succeeding_layer_neuron_strenght = succeeding_layer_neuron_strenght, 
										preceding_neuron_output = preceding_neuron_output, 
										initial_weight_matrix = initial_weight_matrix
										)

		return Array(weight_update_matrix)



	def L2regularizedWeightUpdate(self, learning_rate, delta_n, prev_layer_output, l2_lambda, intial_weight, activation_function = "sigmoid"):
		"""
			Update weight matrix with a regularization method

		"""
		if activation_function == "sigmoid":
			weight_update_matrix = self.sigmoidL2RegularizationWeightUpdate(learning_rate, delta_n, prev_layer_output, l2_lambda, intial_weight)

		return weight_update_matrix



	def getHiddenLayerDelta(self, preceding_neuron_output_arr, weight, proceding_neuron_output_arr, activation_function = "sigmoid"):
		"""
			calculate the strenght of the neurons in a hidden layer 
			
			Arguments:
			preceding_neuron_output (List / Array)		:	The layer of neurons that is first to recieve data relative to forward propagation direction
			weights (List / Array) 						:	The weights in the middle to the two given neurons
			proceding_neuron_strenght (List / Array)	:	The layer of neurons that is second to recieve data relative to forward propagation direction
			
			Retuns: Vector

		"""
		if activation_function == "sigmoid":
			delta_vector = self.sigmoidDeltaCalculation(
								preceding_neuron_output_vector = preceding_neuron_output_arr, 
								weight_matrix = weight, 
								proceding_neuron_output_matrix =preceding_neuron_output_arr
								)
		

		return delta_vector


	# rename from getAdjustedBiasdWeights to adjustBiasWeight
	def adjustBiasWeight(self, neuron_strnght):
		"""
			Calculate bias adjustment
			
			Argumemts:
			neuron_strnght	(List / Array)	:	Updated neuron strenghts

			Formula: -learning_rate * updated_neuron_strenght
			
			Return Array
		"""
		adjusted_biase = Array(neuron_strnght).multiply(self.learning_rate)
		return Array(adjusted_biase)
		



	def getMeanSquaredError(self, ouput, labeld_output):
		"""
			Calculate the mean squared error or cost value

			Arguments;
			ouput (List / Array) 				:	The unlabled output, or the output from the sigmoid function
			labeld_output (List / Array)		:	The labled output
			
			returns : float
			Formula : 1 / lne(ouput) * sum((ouput - labeld_output) ** 2)

		"""

		arr_difference = self.vectorSubtract(ouput, labeld_output)
		squared_arr = self.vectorSquare(arr_difference)

		arr_sum = Array(squared_arr).sum()
		e = 1 / 3 * arr_sum

		return e










class CreateNetwork(ForwardPropagation, BackPropagation):
	def __init__(self, input_size, layer_size_vectors, learning_rate = -0.01, weight_initializer = "xavierweight", regularization_method = "none", l2_penalty = 0.01):
		super().__init__()
		self.learning_rate = learning_rate
		self.input_size = input_size
		self.layer_size_vectors = layer_size_vectors
		self.weight_initializer = weight_initializer

		self.l2_penalty = l2_penalty
		self.regularization_method = regularization_method

		self.layer_sizes = self.initailizeLayerSizes()
		self.weights_set = self.initializeLayerWeights()
		self.bias_weight_set = self.initializeBiasedWeights()

		self.mean_square_error_log = []
		self.batch_array = []
		self.answer_key_batch_array = []
		self.accuracy = 0.0


	
	def fit(self, training_data, labeld_outputs, epoch, batch_size):
		"""
			Arguments: 
			training_data (Matrix)			: Matrix of the training data
			labeld_outputs (Matrix)			: Matrix of the labled output of the training data
			epoch (scalar int)				: The amount of loop i will do to look over the entire training data
			batch_size (scalar int)			: The amount of batches of training data to be trained

		"""
		# print summary of the model
		self.printNetworkPrelimSummary(epoch, batch_size)
		
		# Devide the training data in batches
		self.batch_array, self.answer_key_batch_array = self.devideBatches(training_data, labeld_outputs, batch_size)

		# get the value of how many layers the network have
		layer_count = len(self.layer_sizes)

		# get how many batches is needed to loop through
		batches_count = len(self.batch_array)

		# count the number of correct prediction to calculate accuracy of the model
		correct_prediction = 0


		for _ in tqdm(range(epoch)):
			for training_batch_set_index in range(batches_count):

				# Get the batch of training data by accessing its index from the pool of the bactch
				training_batch = self.batch_array[training_batch_set_index]
				
				# The answer key or labeled ouput of the current batch
				batch_key = self.answer_key_batch_array[training_batch_set_index]



				# loop through the entire data inside the batch 
				for data_index in range(len(training_batch)):
					# get the input data for the current loop
					input_data = training_batch[data_index]
					# get the labeld output of the input data
					input_labeld_data = batch_key[data_index]



					#### FORWARD PROPAGATION ####
					"""
					the varianle "layer_ouputs_matrix" holds the value of the ouputs of the previous layer the training_data variable is added intialy as it would represent as the final ouput for the first layer in back propagation
					"""
					layer_ouputs_matrix = []
					

					# The input of the current layer
					current_layer_input = input_data

					# Loop through the entire layer of the neural network
					for layer_index in range(layer_count):

						layer_activation_function = self.layer_size_vectors[layer_index][1]

						# create a layer where neuron activation and other transformation will handle
						layer_ouput = self.fowardPropagation(
										input_value = current_layer_input, 
										weight_value = self.weights_set[layer_index], 
										bias_weight = self.bias_weight_set[layer_index],
										activation_function = "sigmoid"
									)

						# Append the output of the layer for backpropagation
						layer_ouputs_matrix.append(layer_ouput)
						# update the input for the next layer
						current_layer_input = layer_ouput


					#### calculate Loss functions ####
					mean_square_error = self.getMeanSquaredError(
						 						ouput = layer_ouputs_matrix[-1], 
						 						labeld_output = input_labeld_data
						 						)
					# Append the result to the error log
					self.mean_square_error_log.append(mean_square_error)
					# Check if the model prediction was correct
					final_prediction = ActivationFunction().argMax(layer_ouputs_matrix[-1])
					
					if final_prediction == input_labeld_data:
						correct_prediction += 1




					#### BACK PROPAGATION #####
					# delta of the current layer in loop, Initial value was calculated using the final layer strenght
					delta_h = self.getFinalLayerDelta(
											predicted_ouputs_vector = layer_ouputs_matrix[-1], 
											actual_label_vector = input_labeld_data
										)

					# Loop throught the entire layer from start to the final
					for layer_index in range(len(self.layer_size_vectors) - 1, -1, -1): 
						# get the specific activation function use by the current layer in iteration
						layer_activation_function = self.layer_size_vectors[layer_index][1]
						# update the weight of the biases every layer using the delta_h
						self.bias_weight_set[layer_index] = self.adjustBiasWeight(neuron_strnght = delta_h)

						# check if the layer in proccess is a hidden layer
						if layer_index != 0:
							# if the regularization_method is set to L2 regularization method
							if self.regularization_method == "L2":
								# calculate the adjustment needed to the weight with the L2 regualrization equation
								weight_update = self.L2regularizedWeightUpdate(
																learning_rate = self.learning_rate, 
																delta_n = delta_h, 
																prev_layer_output = layer_ouputs_matrix[layer_index - 1], 
																l2_lambda = self.l2_penalty, 
																intial_weight = self.weights_set[layer_index],
																activation_function = layer_activation_function
															)

							# if there is no regularization method used the the weight is calculated using the sigmoid derivative in calculateWeightAdjustment method
							elif self.regularization_method == "none":
								# calculate the adjustment needed to apply to the initial weight
								weight_update = self.updateLayerWeight(
														succeeding_layer_neuron_strenght = delta_h, 
														preceding_neuron_output = layer_ouputs_matrix[layer_index - 1], 
														initial_weight_matrix = self.weights_set[layer_index]
													)


							# Update the layer weight with the new calculated weight
							self.weights_set[layer_index] = weight_update
							layer_strenght = self.getHiddenLayerDelta(
										preceding_neuron_output_arr = layer_ouputs_matrix[layer_index - 1], 
										weight = weight_update, 
										proceding_neuron_output_arr = delta_h,
										activation_function = "sigmoid"
									)

							# update the value of delta_h coming from the calculated value of the hidden layer strenghts
							delta_h = layer_strenght


						## check if the current layer in interation is the main input layer ##
						elif layer_index == 0:
							if self.regularization_method == "L2":
								weight_update = self.L2regularizedWeightUpdate(
																learning_rate = self.learning_rate, 
																delta_n = delta_h, 
																prev_layer_output = input_data, 
																l2_lambda = self.l2_penalty, 
																intial_weight = self.weights_set[layer_index],
																activation_function = layer_activation_function
															)
								
							elif self.regularization_method == "none":
								weight_update = self.updateLayerWeight(
														succeeding_layer_neuron_strenght = delta_h, 
														preceding_neuron_output = input_data, 
														initial_weight_matrix = self.weights_set[layer_index]
													)

							self.weights_set[layer_index] = weight_update
							break


		# After fitting calculate the model acccuracy
		self.calculateModelAccuracy(
				n_of_correct_pred = correct_prediction, 
				n_of_training_data = len(training_data), 
				training_epoch = epoch
			)

		self.printFittingSummary()



	def predict(self, input_data):
		layer_input = input_data
		layer_output_arr = []

		for layer_index in range(len(self.layer_sizes)):
			# create a layer where neuron activation and other transformation will handle
			layer_ouput = self.fowardPropagation(
							input_value = layer_input, 
							weight_value = self.weights_set[layer_index], 
							bias_weight = self.bias_weight_set[layer_index],
							activation_function = "sigmoid"
							)
			layer_output_arr.append(layer_ouput)
			layer_input = layer_ouput
		
		return layer_output_arr[-1]


	def initailizeLayerSizes(self):
		layer_sizes = []
		for layer_index in range(len(self.layer_size_vectors)):
			current_layer_size = self.layer_size_vectors[layer_index][0]

			if layer_index == 0:
				new_layer = [current_layer_size, self.input_size]
			elif layer_index != 0 :
				new_layer = [current_layer_size, self.layer_size_vectors[layer_index - 1][0]]

			layer_sizes.append(new_layer)
		return layer_sizes



	def initializeLayerWeights(self):
		new_weight_set = []
		for layer_index in range(len(self.layer_sizes)):
			if self.weight_initializer == "xavierweight":
				if layer_index != len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							self.layer_sizes[layer_index], 
							self.layer_sizes[layer_index][0], 
							self.layer_sizes[layer_index + 1][0]
						)
				
				elif layer_index == len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							self.layer_sizes[layer_index], 
							self.layer_sizes[layer_index][0],
							0
						)

			elif self.weight_initializer == "simple":
				new_weight = WeightInitializer().intializeWeight(self.layer_sizes[layer_index])


			new_weight_set.append(new_weight)

		return new_weight_set



	def initializeBiasedWeights(self):
		new_bias_weight_set = []

		for layer_index in range(len(self.layer_sizes)):
			bias_weight_dim = [1, self.layer_sizes[layer_index][0]]

			if self.weight_initializer == "xavierweight":
				if layer_index != len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							bias_weight_dim, 
							self.layer_sizes[layer_index][0], 
							self.layer_sizes[layer_index + 1][0]
						)

				elif layer_index == len(self.layer_sizes) - 1:
					new_weight = WeightInitializer().initNormalizedXavierWeight(
							bias_weight_dim, 
							self.layer_sizes[layer_index][0],
							0
						)


			elif self.weight_initializer == "simple":
				new_weight = WeightInitializer().intializeWeight(bias_weight_dim)

	
			new_bias_weight_set.append(new_weight)

		return new_bias_weight_set



	def devideBatches(self, train_data_arr, answer_key_arr, batch_size):
		test_data_lenght = len(train_data_arr)

		if test_data_lenght < batch_size:
			raise ValueError("Bacth size cannot be grater that the size of the training data")

		test_data_batch_array = []
		answer_key_batch_array = []

		for index in range(0, test_data_lenght, batch_size):
			test_data_batch_array.append(train_data_arr[index: batch_size + index])
			answer_key_batch_array.append(answer_key_arr[index: batch_size + index])

		return test_data_batch_array, answer_key_batch_array


	def printNetworkPrelimSummary(self, epoch, batch_size):
		tab_distance = 4
		tab = "...." * tab_distance
		print("#" * 34, "Network Summary", "#" * 34)
		print("Fitting model with ", epoch, " epoch and ", batch_size, " Batch size")

		print("\nNetwork Architecture: ")
		print("	Learning Rate:", tab, tab, tab, self.learning_rate)
		print("	Regularization:", tab, tab, tab, self.regularization_method)

		if self.regularization_method == "L2":
			print("	L2-Penalty:		", tab, tab, tab,self.l2_penalty)


		print("Layers: ")
		for _layer_index in range(len(self.layer_sizes)):
			print("	Layer: ", _layer_index + 1,  "	Activation Function: Sigmoid Function", tab, self.layer_sizes[_layer_index][0], " Neurons")

		print("\nFitting Progress:")



	def printFittingSummary(self):
		tab_distance = 4
		tab = "...." * tab_distance

		print("\nTraining Complete: ")
		print("Model accuracy: ", self.accuracy)
		print("#" * 34, "End of Summary", "#" * 34)



	def calculateModelAccuracy(self, n_of_correct_pred, n_of_training_data, training_epoch):
		"""
			Calculate the model accurary

			Arguments:
				n_of_correct_pred (scala intiger)			: The number of correct prediction the model made
				n_of_training_data (scalar intiger)			: The total number of the trianing data fed during training
 		"""

		self.accuracy = (n_of_correct_pred / (n_of_training_data * training_epoch)) * 100