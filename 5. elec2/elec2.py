import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import deque
from river import datasets, naive_bayes, tree, linear_model, compose, preprocessing, forest
import csv


TOTAL_ITERATIONS = 20000
dataset = datasets.Elec2()

stream_demand = []
stream_labels = []

# IELM Classifier
class IELMClassifier:
    def __init__(self, n_hidden=64): # n_hidden is the number of neurons in the hidden layer
        self.L = n_hidden # L is the number of hidden neurons
        self.W = None # W is the weight matrix for the hidden layer, initialized later when we see the first sample
        self.b = np.random.randn(n_hidden) # b is the bias for the hidden layer, initialized randomly
        self.beta = np.zeros(n_hidden) # beta is the output weight vector, initialized to zeros
        self.P = None # P is the memory matrix for the online update
        self.C = 1e4 # C is a large constant used in the RLS update to ensure numerical stability

#this function does the online learning for one sample at a time, which is different from the classic ELM that learns in batch mode. It uses recursive least squares (RLS) to update the output weights efficiently without needing to retrain on the entire dataset.
    def learn_one(self, x, y): # x is the input sample (a dictionary), y is the true label for this sample
        xv = np.array(list(x.values()))
        #this fuction ensures the learning one at the time, since classic ELM is a batch learning
        # using recursive least squares (RLS) for online update of output weights
        if self.W is None:
            self.W = np.random.randn(self.L, len(xv)) * np.sqrt(2.0 / len(xv))
            self.P = self.C * np.eye(self.L)
        
        # Hidden layer activation
        h = 1.0 / (1.0 + np.exp(-np.clip(self.W @ xv + self.b, -30, 30))) # what h stands for? hidden layer output, using sigmoid activation function with clipping to prevent overflow
        
# this function does the online update of the output weights (beta) using the current hidden layer output (h) and the true label 
# (y). It uses the Sherman-Morrison formula to efficiently update the memory matrix P and the 
# output weights without needing to invert a large matrix, which is crucial for online learning scenarios.
        # Online update - using Sherman-Morrison formula for efficient matrix inversion
        Ph = self.P @ h  #updates memory matrix P using the current hidden layer output
        self.P = self.P - np.outer(Ph, Ph) / (1.0 + h @ Ph) # updates the output weights beta using the current error (y - h @ beta)
        self.beta = self.beta + self.P @ h * (y - h @ self.beta) # returns the updated model after learning from the current sample
        return self

# this function makes a prediction for a new input sample x by pushing it through the random hidden layer to get the hidden layer output (h) and then computing the final output using the output weights (beta). The prediction is binary based on whether the output is above or below 0.5.
    def predict_one(self, x):  #when new data point (x) arrives, model pushes it through the random hidden layer to get hidden layer output (h) and then computes the final output using the output weights (beta). The prediction is binary based on whether the output is above or below 0.5.
        if self.W is None: return 0
        xv = np.array(list(x.values())) 
        h = 1.0 / (1.0 + np.exp(-np.clip(self.W @ xv + self.b, -30, 30)))
        return int(h @ self.beta >= 0.5)

models = {
    "Gaussian NB": naive_bayes.GaussianNB(),
    "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
    "IELM": IELMClassifier(),
    "SGD Perceptron": compose.Pipeline(preprocessing.StandardScaler(), linear_model.Perceptron()),
    "Adaptive RF": forest.ARFClassifier(n_models=10, seed=42)
}

# training
window_size = 500 
rolling_correct = {name: deque(maxlen=window_size) for name in models}
accuracy_history = {name: [] for name in models}

for i, (x, y) in enumerate(dataset):
    if i >= TOTAL_ITERATIONS:
        break
    y_int = 1 if y else 0

    stream_demand.append(x['nswdemand'])
    stream_labels.append(y_int)
    
    for name, model in models.items():

        y_pred = model.predict_one(x)
        y_pred_int = 1 if y_pred else 0
        
        rolling_correct[name].append(int(y_pred_int == y_int))
        accuracy_history[name].append(np.mean(rolling_correct[name]))
        
        model.learn_one(x, y_int)

stream_demand = np.array(stream_demand)
stream_labels = np.array(stream_labels)


# data steam and streams' data distribution histogram
fig = plt.figure(figsize=(12, 4), tight_layout=True)
gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])

ax1.grid(alpha=0.3)
ax1.plot(stream_demand, label='NSW Power Demand', alpha=0.8, color='tab:blue', linewidth=0.5)
ax1.set_title("NSW Electricity Demand")
ax1.set_xlabel("Iterations")
ax1.set_xlim(0, TOTAL_ITERATIONS)
ax1.legend()

ax2.grid(axis='y', alpha=0.3)
ax2.hist(stream_demand[stream_labels == 0], bins=30, alpha=0.5, label='Price DOWN (0)', color='tab:orange')
ax2.hist(stream_demand[stream_labels == 1], bins=30, alpha=0.5, label='Price UP (1)', color='tab:purple')
ax2.set_title("Demand Distributions by Class")
ax2.legend()
        
plt.savefig('elec2_distribution.png')
# plt.show()

plt.figure(figsize=(12, 6))

for name, history in accuracy_history.items():
    plt.plot(history, label=f"{name} (Final: {history[-1]:.2f})", linewidth=1.5)

plt.title("Classification Accuracy on Real-World Data (Elec2)")
plt.xlabel("Iterations")
plt.ylabel(f"Rolling Accuracy (Window={window_size})")
plt.xlim(0, TOTAL_ITERATIONS)
plt.ylim(0.0, 1.0)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('elec2_accuracies.png')
# plt.show()

with open('elec2_accuracies.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    model_names = list(accuracy_history.keys())
    header = ["Iteration"] + model_names
    writer.writerow(header)
    
    for i in range(TOTAL_ITERATIONS):
        row = [i + 1]
        for name in model_names:
            row.append(f"{accuracy_history[name][i]:.4f}")
        writer.writerow(row)