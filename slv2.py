import matplotlib.pyplot as plt

# Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(loss_curve)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Learning Rate Schedule
plt.figure(figsize=(10, 5))
plt.plot(learning_rate_schedule)
plt.title('Learning Rate Schedule')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.show()

# Activation Histograms
activation_data = model.activations(input_tensor)  # Assuming you have a method to get activations
plt.figure(figsize=(10, 5))
plt.hist(activation_data.flatten(), bins=50)
plt.title('Activation Histogram')
plt.xlabel('Activation Value')
plt.ylabel('Frequency')
plt.show()

# Weight Histograms
plt.figure(figsize=(10, 5))
plt.hist(model.fc.weight.detach().numpy().flatten(), bins=50)
plt.title('Weight Histogram')
plt.xlabel('Weight Value')
plt.ylabel('Frequency')
plt.show()

# Parameter Updates
plt.figure(figsize=(10, 5))
plt.plot(parameter_updates)
plt.title('Parameter Updates')
plt.xlabel('Iteration')
plt.ylabel('Update Magnitude')
plt.show()

# Gradient Norms
plt.figure(figsize=(10, 5))
plt.plot(gradient_norms)
plt.title('Gradient Norms')
plt.xlabel('Iteration')
plt.ylabel('L2 Norm')
plt.show()

# Validation Metrics
plt.figure(figsize=(10, 5))
plt.plot(validation_metrics)
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Metric Value')
plt.show()

# Activations Over Time
plt.figure(figsize=(10, 5))
plt.plot(activations_over_time)
plt.title('Activations Over Time')
plt.xlabel('Time Step')
plt.ylabel('Activation Value')
plt.show()
