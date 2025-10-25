# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Image data: 5x5 pixels, 3-bit (values 0-7)
data = [5,3,1,0,1,0,2,1,0,5,1,5,0,1,2,4,2,6,2,1,6,2,0,1,5]
image = np.array(data).reshape(5, 5)

print("="*60)
print("Task 3: Histogram Equalization - Manual Calculation")
print("="*60)
print("\n[Original Image (5x5)]:")
print(image)

# Parameters
M, N = 5, 5  # Image size
L = 8        # Number of intensity levels (3 bits = 2^3 = 8 levels: 0-7)
total_pixels = M * N  # 25 pixels

print(f"\nParameters:")
print(f"  - Image size: {M}x{N} = {total_pixels} pixels")
print(f"  - Data size: 3 bits -> L = {L} levels (0-7)")

# ============================================
# Step 1: Create Histogram
# ============================================
print("\n" + "="*60)
print("Step 1: Create Histogram (Count pixels at each level)")
print("="*60)

# Count pixels at each intensity level
histogram = {}
for i in range(L):
    histogram[i] = np.sum(image == i)

# Create table
df_hist = pd.DataFrame({
    'r_k (Intensity)': list(histogram.keys()),
    'n_k (Count)': list(histogram.values())
})

print(df_hist.to_string(index=False))

# ============================================
# Step 2: Calculate probability p_r(r_k)
# ============================================
print("\n" + "="*60)
print("Step 2: Calculate probability p_r(r_k) = n_k / MN")
print("="*60)

probability = {}
for i in range(L):
    probability[i] = histogram[i] / total_pixels

df_hist['p_r(r_k)'] = [probability[i] for i in range(L)]
print(df_hist.to_string(index=False))

# ============================================
# Step 3: Calculate CDF (Cumulative Distribution Function)
# ============================================
print("\n" + "="*60)
print("Step 3: Calculate CDF (Cumulative Distribution Function)")
print("="*60)
print("Formula: cdf(r_k) = sum(j=0 to k) p_r(r_j)")

cdf = {}
cumsum = 0
for i in range(L):
    cumsum += probability[i]
    cdf[i] = cumsum

df_hist['CDF'] = [cdf[i] for i in range(L)]
print(df_hist.to_string(index=False))

# ============================================
# Step 4: Calculate new values s_k
# ============================================
print("\n" + "="*60)
print("Step 4: Calculate new intensity values s_k")
print("="*60)
print(f"Formula: s_k = T(r_k) = (L-1) x CDF(r_k) = {L-1} x CDF(r_k)")
print(f"         s_k = floor({L-1} x CDF(r_k))")

# Calculate new values
s_k_continuous = {}
s_k_rounded = {}

for i in range(L):
    s_k_continuous[i] = (L - 1) * cdf[i]
    s_k_rounded[i] = int(np.floor(s_k_continuous[i]))

df_hist['(L-1)xCDF'] = [s_k_continuous[i] for i in range(L)]
df_hist['s_k = floor()'] = [s_k_rounded[i] for i in range(L)]

print("\n" + df_hist.to_string(index=False))

# ============================================
# Step 5: Create Mapping Table
# ============================================
print("\n" + "="*60)
print("Step 5: Mapping Table (r_k -> s_k)")
print("="*60)

mapping_table = pd.DataFrame({
    'Original (r_k)': list(range(L)),
    'New (s_k)': [s_k_rounded[i] for i in range(L)]
})
print(mapping_table.to_string(index=False))

# ============================================
# Step 6: Transform Image
# ============================================
print("\n" + "="*60)
print("Step 6: Transform Image using Mapping Table")
print("="*60)

# Create new image
equalized_image = np.zeros_like(image)
for i in range(M):
    for j in range(N):
        original_value = image[i, j]
        equalized_image[i, j] = s_k_rounded[original_value]

print("\n[Original Image]:")
print(image)
print("\n[Equalized Image]:")
print(equalized_image)

# ============================================
# Step 7: Show results as 1D array
# ============================================
print("\n" + "="*60)
print("Step 7: Intensity values for each pixel (1D Array)")
print("="*60)

print("\n[Original pixels]:")
print(image.flatten())
print("\n[Equalized pixels]:")
print(equalized_image.flatten())

# ============================================
# Visualization
# ============================================
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original Image
axes[0, 0].imshow(image, cmap='gray', vmin=0, vmax=7)
axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')
for i in range(5):
    for j in range(5):
        axes[0, 0].text(j, i, str(image[i, j]), 
                       ha='center', va='center', color='red', fontsize=14)

# Original Histogram
axes[0, 1].bar(list(histogram.keys()), list(histogram.values()), 
               color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Intensity Level (r)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Original Histogram', fontsize=12, fontweight='bold')
axes[0, 1].set_xticks(range(L))
axes[0, 1].grid(axis='y', alpha=0.3)

# CDF Plot
axes[0, 2].plot(list(cdf.keys()), list(cdf.values()), 
                marker='o', linewidth=2, markersize=8, color='green')
axes[0, 2].set_xlabel('Intensity Level (r)')
axes[0, 2].set_ylabel('CDF')
axes[0, 2].set_title('Cumulative Distribution Function', fontsize=12, fontweight='bold')
axes[0, 2].set_xticks(range(L))
axes[0, 2].grid(True, alpha=0.3)

# Equalized Image
axes[1, 0].imshow(equalized_image, cmap='gray', vmin=0, vmax=7)
axes[1, 0].set_title('Equalized Image', fontsize=12, fontweight='bold')
axes[1, 0].axis('off')
for i in range(5):
    for j in range(5):
        axes[1, 0].text(j, i, str(equalized_image[i, j]), 
                       ha='center', va='center', color='red', fontsize=14)

# Equalized Histogram
eq_histogram = {}
for i in range(L):
    eq_histogram[i] = np.sum(equalized_image == i)

axes[1, 1].bar(list(eq_histogram.keys()), list(eq_histogram.values()), 
               color='orange', alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('Intensity Level (s)')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Equalized Histogram', fontsize=12, fontweight='bold')
axes[1, 1].set_xticks(range(L))
axes[1, 1].grid(axis='y', alpha=0.3)

# Mapping Visualization
axes[1, 2].plot(list(s_k_rounded.keys()), list(s_k_rounded.values()), 
                marker='o', linewidth=2, markersize=10, color='red')
axes[1, 2].set_xlabel('Original Intensity (r)')
axes[1, 2].set_ylabel('New Intensity (s)')
axes[1, 2].set_title('Transformation Function T(r)', fontsize=12, fontweight='bold')
axes[1, 2].set_xticks(range(L))
axes[1, 2].set_yticks(range(L))
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('histogram_equalization_manual.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("[SUCCESS] Histogram Equalization Complete!")
print("[SAVED] Image saved as: histogram_equalization_manual.png")
print("="*60)