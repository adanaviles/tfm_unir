import pickle

import matplotlib.pyplot as plt
import xgboost as xgb

xgb_model = pickle.load(open("xgboost_model_gs.pkl", "rb"))


# Set the figure size
plt.figure(figsize=(15, 15))

# Plot feature importance and get the Axes object
ax = xgb.plot_importance(xgb_model, importance_type="gain", max_num_features=20, values_format="{v:.2f}")
plt.title("Feature Importance (gain) xgboost")

# Ensure the layout fits the feature names
plt.tight_layout()

# Show the plot
plt.show()

# Set the figure size
plt.figure(figsize=(15, 15))

# Plot feature importance and get the Axes object
ax = xgb.plot_importance(xgb_model, importance_type="weight", max_num_features=20, values_format="{v:.2f}")
plt.title("Feature Importance (weight) xgboost")

# Ensure the layout fits the feature names
plt.tight_layout()

# Show the plot
plt.show()
