import cv2
import numpy as np
from matplotlib import pyplot as plt

# Read image
img = cv2.imread('inside-the-box.jpg', 0)  # 0 = grayscale

if img is None:
    print("Error: Cannot read image file. Please check the path.")
    exit()

# ============================================
# Technique 1: Histogram Equalization
# ============================================
img_histeq = cv2.equalizeHist(img)

# ============================================
# Technique 2: Log Transformation
# ============================================
# Formula: s = c * log(1 + r)
c = 255 / np.log(1 + np.max(img))
img_log = c * np.log(1 + img.astype(float))
img_log = np.array(img_log, dtype=np.uint8)

# ============================================
# Technique 3: Gamma Transformation (0.2)
# ============================================
gamma = 0.2
img_gamma = np.array(255 * (img / 255) ** gamma, dtype=np.uint8)

# ============================================
# Technique 4: Contrast Stretching
# ============================================
p2, p98 = np.percentile(img, (2, 98))
img_contrast = np.clip((img - p2) * (255 / (p98 - p2)), 0, 255).astype(np.uint8)

# ============================================
# Technique 5: Bit-plane Slicing (Bit 0-3)
# ============================================
def extract_bit_plane(img, bit):
    """Extract specific bit plane from image"""
    bit_plane = (img >> bit) & 1
    return bit_plane * 255

# Extract bit planes 0-3 and reconstruct image
bit_planes = []
reconstructed_low = np.zeros_like(img)

for i in range(4):
    bit_plane = extract_bit_plane(img, i)
    bit_planes.append(bit_plane)
    # Reconstruct by adding each bit plane with proper weight
    reconstructed_low += (bit_plane // 255) * (2 ** i)

reconstructed_low = reconstructed_low.astype(np.uint8)

# ============================================
# Display Results
# ============================================
fig = plt.figure(figsize=(12, 8))

# Row 1: Enhancement Techniques
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray')
plt.title('Original Image', fontweight='bold', fontsize=13, pad=20)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_histeq, cmap='gray')
plt.title('1. Histogram Equalization', fontweight='bold', fontsize=12, pad=20)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_log, cmap='gray')
plt.title('2. Log Transformation', fontweight='bold', fontsize=12, pad=20)
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(img_gamma, cmap='gray')
plt.title(f'3. Gamma = {gamma}', fontweight='bold', fontsize=12, pad=20)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img_contrast, cmap='gray')
plt.title('4. Contrast Stretching', fontweight='bold', fontsize=12, pad=20)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(reconstructed_low, cmap='gray')
plt.title('5. Bit-planes 0-3', fontweight='bold', fontsize=12, pad=20)
plt.axis('off')

plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.3, wspace=0.1)

plt.tight_layout()
plt.savefig('inside_box_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================
# Save Result Images
# ============================================
cv2.imwrite('result_histogram_equalization.jpg', img_histeq)
cv2.imwrite('result_log_transformation.jpg', img_log)
cv2.imwrite('result_gamma_0.2.jpg', img_gamma)
cv2.imwrite('result_contrast_stretching.jpg', img_contrast)
cv2.imwrite('result_bitplane_0-3_combined.jpg', reconstructed_low)

# Save individual bit planes for reference
for i in range(4):
    cv2.imwrite(f'result_bitplane_{i}.jpg', bit_planes[i])

print("="*60)
print("[SUCCESS] Image enhancement complete!")
print("="*60)
print("\nTechniques used:")
print("  1. Histogram Equalization")
print("  2. Log Transformation")
print("  3. Gamma Transformation (gamma = 0.2)")
print("  4. Contrast Stretching")
print("  5. Bit-plane Slicing (Bit 0-3 combined)")
print("\nFiles saved:")
print("  - inside_box_results.png (all results)")
print("  - result_bitplane_0-3_combined.jpg (main bit-plane result)")
print("  - result_*.jpg (individual results)")
print("="*60)