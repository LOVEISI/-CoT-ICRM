import matplotlib.pyplot as plt

# Data
steps_new = range(0, 2001, 50)  # Assuming 40 steps (0-1950)
accuracy_new = [
    0.2595, 0.8455, 0.9053, 0.9302, 0.9398, 0.9484, 0.9599, 0.9651, 0.9675, 0.9720,
    0.9739, 0.9756, 0.9783, 0.9815, 0.9815, 0.9815, 0.9829, 0.9844, 0.9865, 0.9870,
    0.9878, 0.9881, 0.9887, 0.9892, 0.9892, 0.9895, 0.9900, 0.9913, 0.9913, 0.9913,
    0.9920, 0.9924, 0.9924, 0.9930, 0.9933, 0.9933, 0.9933, 0.9933, 0.9938, 0.9946, 0.9953
]

steps_original = range(0, 2001, 50)  # Original steps (0-1100)
accuracy_original = [
    0.4906, 0.7670, 0.8297, 0.8469, 0.8891, 0.9156, 0.9414, 0.9414, 0.9527, 0.9527,
    0.9600, 0.9600, 0.9600, 0.9600, 0.9631, 0.9644, 0.9644, 0.9711, 0.9711, 0.9711,
    0.9711, 0.9743, 0.9750, 0.9764, 0.9764, 0.9764, 0.9768, 0.9783, 0.9783, 0.9783,
    0.9783, 0.9792, 0.9792, 0.9810, 0.9821, 0.9821, 0.9821, 0.9836, 0.9836, 0.9836, 0.9836
]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(steps_new[:len(accuracy_new)], accuracy_new, label='w FEAT', marker='o')
plt.plot(steps_original[:len(accuracy_original)], accuracy_original, label='w/o FEAT', marker='x')
plt.xlabel('Training Steps (per 50)')
plt.ylabel('best_te Accuracy')
plt.title('Comparison of Accuracy')
plt.legend()
plt.grid(True)
plt.show()