import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torchvision import models

def compute_erf(model, input_size=512, num_iterations=20):
    model.eval()
    accumulated_grad = torch.zeros((input_size, input_size))
    
    print(f"Averaging ERF over {num_iterations} iterations...")

    for i in range(num_iterations):
        # 1. Create a random input (Gaussian noise)
        input_image = torch.randn(1, 3, input_size, input_size, requires_grad=True)
        
        # 2. Forward pass (extracting layer4 specifically for ResNet)
        # Using the same path as before, but slightly more condensed
        x = model.conv1(input_image)
        x = model.bn1(x)
        x = model.relu(x)
        x = model.maxpool(x)
        x = model.layer1(x)
        x = model.layer2(x)
        x = model.layer3(x)
        features = model.layer4(x)
        
        # 3. Isolate central unit
        _, c, h, w = features.shape
        central_unit = features[0, :, h//2, w//2].sum()
        
        # 4. Backward pass
        model.zero_grad()
        central_unit.backward()
        
        # 5. Accumulate the absolute gradient magnitude
        # We take the absolute value so that negative and positive influence don't cancel out
        grad = input_image.grad.detach().cpu()[0]
        grad_abs = torch.abs(grad).mean(dim=0) # Average over channels
        accumulated_grad += grad_abs
        
    # 6. Final processing
    # Average and normalize to [0, 1]
    erf_map = accumulated_grad / num_iterations
    erf_map = (erf_map - erf_map.min()) / (erf_map.max() - erf_map.min())
    
    return erf_map.numpy()

# Initialize Model
resnet = models.resnet50(pretrained=True)

# Run Measurement
final_erf = compute_erf(resnet, input_size=512, num_iterations=100)

# 7. Visualization with a 'log' scale option
# ERF values often drop off exponentially, so log scale helps see the outer edges
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(final_erf, cmap='viridis')
plt.title("Linear Scale ERF")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(np.log(final_erf + 1e-7), cmap='viridis')
plt.title("Log Scale ERF (Shows boundaries better)")
plt.colorbar()

plt.show()

def calculate_erf_statistics(erf_map, threshold=0.95):
    # 1. Sort all pixel values in descending order
    flat_erf = erf_map.flatten()
    sorted_erf = np.sort(flat_erf)[::-1]
    
    # 2. Find the cumulative sum and the cutoff value for the threshold
    cum_sum = np.cumsum(sorted_erf)
    total_sum = cum_sum[-1]
    cutoff_idx = np.where(cum_sum >= total_sum * threshold)[0][0]
    cutoff_value = sorted_erf[cutoff_idx]
    
    # 3. Create a mask of pixels that contribute to the top X%
    mask = erf_map >= cutoff_value
    
    # 4. Find the bounding box of these pixels
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    height = y_max - y_min
    width = x_max - x_min
    
    return (height, width), mask

# Calculate stats for our previously computed final_erf
(h, w), impact_mask = calculate_erf_statistics(final_erf, threshold=0.95)

print(f"ERF 95% Bounding Box: {h}x{w} pixels")
print(f"Effective Diameter (Avg): {(h + w) / 2:.2f} pixels")

# Visualize the 'Active' Zone
plt.imshow(impact_mask, cmap='gray')
plt.title(f"95% Impact Zone ({h}x{w})")
plt.show()