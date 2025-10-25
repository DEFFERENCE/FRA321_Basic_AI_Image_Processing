import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('mega_space_molly.jpg')

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Unsharp Masking Function
def unsharp_mask(image, kernel_size=5, sigma=3, k=1):
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    mask = cv2.subtract(image, blurred)
    result = cv2.addWeighted(image, 1, mask, k, 0)
    return np.clip(result, 0, 255).astype(np.uint8)

# Parameters
kernel_size, sigma = 5, 3

# Apply filters with k = 1, 2, 10
result_k1 = unsharp_mask(img_rgb, kernel_size, sigma, k=1)
result_k2 = unsharp_mask(img_rgb, kernel_size, sigma, k=2)
result_k10 = unsharp_mask(img_rgb, kernel_size, sigma, k=10)

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Display images
images = [img_rgb, result_k1, result_k2, result_k10]
titles = ['Original', 
          'Unsharp Masking (k=1)', 
          'High-boost Filtering (k=2)', 
          'High-boost Filtering (k=10)']

for ax, img_data, title in zip(axes.flat, images, titles):
    ax.imshow(img_data)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
    ax.axis('off')

plt.tight_layout()
plt.savefig('task4_results.png', dpi=300, bbox_inches='tight')

# Save individual results
cv2.imwrite('result_original.jpg', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
cv2.imwrite('result_k1.jpg', cv2.cvtColor(result_k1, cv2.COLOR_RGB2BGR))
cv2.imwrite('result_k2.jpg', cv2.cvtColor(result_k2, cv2.COLOR_RGB2BGR))
cv2.imwrite('result_k10.jpg', cv2.cvtColor(result_k10, cv2.COLOR_RGB2BGR))

plt.show()
