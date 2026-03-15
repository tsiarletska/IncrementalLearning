import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from collections import deque
from river import naive_bayes, tree, linear_model, compose, preprocessing

import csv


random_state = np.random.RandomState(42)
TOTAL_ITERATIONS = 20000

stream_values = []
stream_labels= []
drifts = []
segment_lengths = []
segment_colors = []

while len(stream_values) < TOTAL_ITERATIONS:
    Ri = random_state.randint(-50, 51)
    Ti = 2000 + Ri
    
    if len(stream_values) + Ti > TOTAL_ITERATIONS:
        Ti = TOTAL_ITERATIONS - len(stream_values)

    choice = random_state.randint(1, 6)
    
    if choice == 1:
        mu, sigma, direction, color = 0.5, 0.4, "right", "tab:blue"
    elif choice == 2:
        mu, sigma, direction, color = 1.0, 0.3, "left",  "tab:orange" 
    elif choice == 3:
        mu, sigma, direction, color = 0.8, 0.5, "right", "tab:green"
    elif choice == 4:
        mu, sigma, direction, color = 1.5, 0.2, "left",  "tab:red" 
    elif choice == 5:
        mu, sigma, direction, color = 0.2, 0.6, "right", "tab:purple"

    base_vals = random_state.lognormal(mean=mu, sigma=sigma, size=Ti)
    base_median = np.exp(mu)
    
    # DYNAMIC RULE: Flip the data and the rule if the direction is "left"
    if direction == "right":
        segment_vals = base_vals
        threshold = base_median
        # Class 1 is the long right tail (greater than median)
        segment_lbls = (segment_vals > threshold).astype(int)
    else:
        # Mirror the data by subtracting it from 10.0
        segment_vals = 10.0 - base_vals
        threshold = 10.0 - base_median
        # Class 1 is now the long left tail! Notice the '<' sign!
        segment_lbls = (segment_vals < threshold).astype(int)

    stream_values.extend(segment_vals)
    stream_labels.extend(segment_lbls)
    
    segment_lengths.append(Ti)
    segment_colors.append(color)
    
    if len(stream_values) < TOTAL_ITERATIONS:
        drifts.append(len(stream_values))


stream_labels = np.array(stream_labels)


print("\n" + "="*40)
print("Concept drift locations")
print("="*40)
print(f"Total drifts generated: {len(drifts)}")
for i, drift_step in enumerate(drifts):
    print(f"  -> Drift {i + 1} at iteration: {drift_step}")
print("="*40 + "\n")

# data steam and streams' data distribution histogram
def plot_stream_and_hist(stream, drifts, segment_lengths, segment_colors, random_state):
    fig = plt.figure(figsize=(12, 4), tight_layout=True)
    gs = gridspec.GridSpec(1, 2, width_ratios=[3, 1])
    ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1])
    
    ax1.grid(alpha=0.3)
    ax1.plot(stream, label='Data Stream', alpha=0.8, color='#333333', linewidth=0.5)
    
    for drift in drifts:
        ax1.axvline(drift, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
        
    start = 0
    for length, color in zip(segment_lengths, segment_colors):
        ax1.axvspan(start, start + length, alpha=0.3, color=color)
        start += length
        
    ax1.set_title("Data Stream with Concept Drifts")
    ax1.set_xlabel("Iterations")
    ax1.set_xlim(0, TOTAL_ITERATIONS)
    
    ax2.grid(axis='y', alpha=0.3)
    dist_info = [
        (0.5, 0.4, "right", "tab:blue", "Dist A (Right)"),
        (1.0, 0.3, "left",  "tab:orange", "Dist B (Left)"),
        (0.8, 0.5, "right", "tab:green", "Dist C (Right)"),
        (1.5, 0.2, "left",  "tab:red", "Dist D (Left)"),
        (0.2, 0.6, "right", "tab:purple", "Dist E (Right)")
    ]
    
    for mu, sigma, direction, color, label in dist_info:
        base_dist = random_state.lognormal(mu, sigma, 2000)
        
        if direction == "right":
            ideal_dist = base_dist
        else:
            ideal_dist = 10.0 - base_dist
            
        ax2.hist(ideal_dist, bins=30, alpha=0.5, label=label, color=color)
        
    ax2.set_title("Mixed Asymmetric Distributions")
    ax2.legend(loc="upper left", fontsize='small')
            
    plt.savefig('drift_detection.png')
    # plt.show()

plot_stream_and_hist(stream_values, drifts, segment_lengths, segment_colors, random_state)

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
    "Naive Bayes": naive_bayes.GaussianNB(),
    "Hoeffding Tree": tree.HoeffdingTreeClassifier(),
    "IELM": IELMClassifier(),
    "SGD Perceptron": compose.Pipeline(preprocessing.StandardScaler(), linear_model.Perceptron())
}

# training
window_size = 500 
rolling_correct = {name: deque(maxlen=window_size) for name in models}
accuracy_history = {name: [] for name in models}

for i in range(TOTAL_ITERATIONS):
    x = {"value": stream_values[i]}
    y = stream_labels[i]
    
    for name, model in models.items():
        y_pred = model.predict_one(x)
        rolling_correct[name].append(int(y_pred == y))
        accuracy_history[name].append(np.mean(rolling_correct[name]))
        
        model.learn_one(x, y)


plt.figure(figsize=(12, 6))

start = 0
for length, color in zip(segment_lengths, segment_colors):
    plt.axvspan(start, start + length, alpha=0.15, color=color)
    start += length

for name, history in accuracy_history.items():
    plt.plot(history, label=f"{name} (Final: {history[-1]:.2f})", linewidth=1.5)

for drift in drifts:
    plt.axvline(drift, color='red', linestyle='--', alpha=0.9, linewidth=1.5)

plt.title("Classification Accuracy Over 20,000 Iterations")
plt.xlabel("Iterations")
plt.ylabel(f"Rolling Accuracy (Window={window_size})")
plt.xlim(0, TOTAL_ITERATIONS)
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.savefig('model_accuracies.png')
# plt.show()


with open('model_accuracies.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    model_names = list(accuracy_history.keys())
    header = ["Iteration"] + model_names
    writer.writerow(header)
    
    for i in range(TOTAL_ITERATIONS):
        row = [i + 1]
        
        for name in model_names:
            row.append(f"{accuracy_history[name][i]:.4f}")
            
        writer.writerow(row)