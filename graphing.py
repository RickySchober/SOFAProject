import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv("test_data2.csv")  # Replace with your actual filename

# Extract relevant columns
time = df["time"]
pos_y = df["pos_y"]
des_y = df["des_y"]
rmse = df[" RMSE"]  # Note the space in the header if it exists

# Plot Y position over time
plt.figure(figsize=(10, 6))
plt.plot(time, pos_y, label="Actual Y Position", linewidth=2)
plt.plot(time, des_y, label="Desired Y Position", linewidth=3)
plt.xlabel("Time (s)")
plt.ylabel("Y Position")
plt.title("Actual vs Desired Y Position Over Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Optionally, plot RMSE over time
plt.figure(figsize=(10, 4))
plt.plot(time, rmse, label="RMSE", color='red')
plt.xlabel("Time (s)")
plt.ylabel("RMSE")
plt.title("RMSE Over Time")
plt.grid(True)
plt.tight_layout()
plt.show()
