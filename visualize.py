import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Iris.csv (1).csv")

# Create pairplot
sns.pairplot(data, hue="Species")

# Show graph
plt.show()