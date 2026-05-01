import numpy as np
import matplotlib.pyplot as plt

# Generate random data
x = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = np.sin(x) + 0.5 * np.random.randn(100)  # sine wave with some noise

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y, '-o', label='Random Data')
plt.title('Random Plot')
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
