import NeuroPy as npy
import pprint
import pandas as pd


pp = pprint.PrettyPrinter(width=41, compact=True)
data = pd.read_csv("IRIS.csv");


print(data.head());



hidden_layers = [(3, "sigmoid"), (2, "sigmoid"), (2, "sigmoid")]

model = npy.CreateNetwork(
		input_size = 3,  
        layer_size_vectors = hidden_layers, 
		learning_rate = -0.01, 
		weight_initializer = "simple", 
		regularization_method = "none", 
		l2_penalty = 0.01
	)


sample_data = [
				[1, 1, 1],
				[0, 0, 0],
			]

labeld_data = [
				[1, 0],
				[0, 1],
			]


# fit the data
result = model.fit(sample_data, labeld_data, 45, 1)
print("prediction: ", model.predict(sample_data[0]))



import matplotlib.pyplot as plt 
x = [i for i in range(len(model.mean_square_error_log))]
y = model.mean_square_error_log
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()