import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('score-results-UPDATEMYNAME.csv', skiprows=1)
plt.plot(data["Num Steps"], data["Train KlDiv"], label = "Training loss")
plt.plot(data["Num Steps"], data["Mean 1-abs(error)"], label = "Validation Accuracy")
plt.legend()
plt.title("Loss over time")
plt.xlabel("Steps")
plt.show()


#### Plot Accuracies
acc_data = pd.read_csv('accuracy.csv')
model_pairs = [("Random_Model_Accuracy", "Random_True_Accuracy"),
               ("Upwards_Model_Accuracy", "Upwards_True_Accuracy"),
               ("Argmax_Model_Accuracy", "Argmax_True_Accuracy")]

# Accuracies for each model pair
accuracies_model = [acc_data["Random_Model_Accuracy"][0], acc_data["Upwards_Model_Accuracy"][0], acc_data["Argmax_Model_Accuracy"][0]]
accuracies_true = [acc_data["Random_True_Accuracy"][0], acc_data["Upwards_True_Accuracy"][0], acc_data["Argmax_True_Accuracy"][0]]

# Set the width of the bars
bar_width = 0.35

# Set the positions of the bars on the x-axis
r1 = np.arange(len(model_pairs))
r2 = [x + bar_width for x in r1]

# Plotting the grouped bar chart
plt.bar(r1, accuracies_model, color='b', width=bar_width, edgecolor='grey', label='Model')
plt.bar(r2, accuracies_true, color='g', width=bar_width, edgecolor='grey', label='True')

# Adding labels
plt.xlabel('Model Pairs', fontweight='bold')
plt.ylabel('Accuracy', fontweight='bold')
plt.xticks([r + bar_width/2 for r in range(len(model_pairs))], [f'{pair[0]} vs {pair[1]}' for pair in model_pairs])

# Adding a legend
plt.legend()

# Showing the plot
plt.show()



###
labels = ['a','b','c','d','e','f']
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, accuracies_model, width, label='Group 1')
rects2 = ax.bar(x + width/2, accuracies_true, width, label='Group 2')


### attempt3
import matplotlib.pyplot as plt
import numpy as np

acc_data = pd.read_csv('accuracy.csv')
sampling = ("Random", "Upwards", "Argmax")
penguin_means = {
    'Model': (acc_data["Random_Model_Accuracy"][0], acc_data["Upwards_Model_Accuracy"][0], acc_data["Argmax_Model_Accuracy"][0]),
    'True': (acc_data["Random_True_Accuracy"][0], acc_data["Upwards_True_Accuracy"][0], acc_data["Argmax_True_Accuracy"][0]),
}

x = np.arange(len(sampling))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots(layout='constrained')

for probDistSource, accuracy in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, accuracy, width, label=probDistSource)
    ax.bar_label(rects, padding=3)
    multiplier += 1

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Length (mm)')
ax.set_title('Penguin attributes by species')
ax.set_xticks(x + width, sampling)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()