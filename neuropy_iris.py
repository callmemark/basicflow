import NeuroPy as npy
import pandas as pd
import matplotlib.pyplot as plt



"""
    Read and preprocess the data
"""
data = pd.read_csv("IRIS.csv");
def convert_label_to_integer(column):
    unique_labels = column.unique()
    label_mapping = {label: i+1 for i, label in enumerate(unique_labels)}
    return column.map(label_mapping)

def one_hot_encode(column):
    encoded_data = pd.get_dummies(column, dtype=int)
    return encoded_data.values

label_transformed = convert_label_to_integer(data['species'])
onehot_encoded = one_hot_encode(data['species'])

def normalize_data(data):
    normalized_data = (data - data.min()) / (data.max() - data.min())
    return normalized_data


training_data = normalize_data(data.iloc[:, 0:4].values)
sample_data = training_data.tolist()
labeld_data	= onehot_encoded.tolist()




"""
    Create and train the model
"""
hidden_layers = [(3, "sigmoid"), (9, "sigmoid"), (9, "sigmoid"), (3, "sigmoid")]
model = npy.CreateNetwork(
		input_size = 4,  
        layer_size_vectors = hidden_layers, 
		learning_rate = -0.01, 
		weight_initializer = "simple", 
		regularization_method = "none", 
		l2_penalty = 0.01
	)

result = model.fit(sample_data, labeld_data, 10, 1)




"""
    Plot the mean square error
"""
x = [i for i in range(len(model.mean_square_error_log))]
y = model.mean_square_error_log
plt.title("Mean square error") 
plt.xlabel("Epoch") 
plt.ylabel("Error value")
plt.plot(x, y) 
plt.show()