import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image
img_path = r'D:\Fibo\Year3\FRA321_Basic_AI\AJ.So\5f8e9ac9bd4879d8794c80436114f6f2text_frombook.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Binarize the image (threshold)
_, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Invert if needed (text should be white on black background)
if np.mean(binary) > 127:
    binary = cv2.bitwise_not(binary)

# Method 1: Using Hole Filling to detect letters with holes
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# Step 1: Fill holes using morphological reconstruction
h, w = binary.shape
marker = np.zeros((h, w), dtype=np.uint8)
marker = cv2.bitwise_not(binary)
marker[1:h-1, 1:w-1] = 0

# Morphological reconstruction by dilation
while True:
    prev = marker.copy()
    marker = cv2.dilate(marker, kernel)
    marker = cv2.bitwise_and(marker, cv2.bitwise_not(binary))
    if np.array_equal(marker, prev):
        break

filled = cv2.bitwise_not(marker)
holes = cv2.subtract(filled, binary)

# Clean noise
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
holes_cleaned = cv2.morphologyEx(holes, cv2.MORPH_OPEN, kernel_open)

# Find contours of letters (for analysis)
letter_contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find contours of holes
hole_contours, _ = cv2.findContours(holes_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create result images
result = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
all_holes_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

# Advanced filtering to separate 'O' from 'p', 'g', 'd'
o_locations = []
other_holes = []

for hole_cnt in hole_contours:
    hole_area = cv2.contourArea(hole_cnt)
    if hole_area < 30:  # Skip very small holes
        continue
    
    # Get hole properties
    hole_x, hole_y, hole_w, hole_h = cv2.boundingRect(hole_cnt)
    hole_center_x = hole_x + hole_w // 2
    hole_center_y = hole_y + hole_h // 2
    
    # Find the parent letter contour
    parent_letter = None
    for letter_cnt in letter_contours:
        # Check if hole center is inside this letter
        if cv2.pointPolygonTest(letter_cnt, (hole_center_x, hole_center_y), False) >= 0:
            parent_letter = letter_cnt
            break
    
    if parent_letter is None:
        continue
    
    # Get letter properties
    letter_x, letter_y, letter_w, letter_h = cv2.boundingRect(parent_letter)
    letter_area = cv2.contourArea(parent_letter)
    
    # Calculate features to distinguish 'O' from 'p', 'g', 'd'
    
    # 1. Hole to letter area ratio (O has larger ratio)
    hole_ratio = hole_area / letter_area if letter_area > 0 else 0
    
    # 2. Vertical position of hole in letter (O's hole is centered)
    hole_relative_y = (hole_center_y - letter_y) / letter_h if letter_h > 0 else 0
    
    # 3. Horizontal position of hole (O's hole is centered)
    hole_relative_x = (hole_center_x - letter_x) / letter_w if letter_w > 0 else 0
    
    # 4. Letter aspect ratio (O is more circular)
    letter_aspect = letter_w / letter_h if letter_h > 0 else 0
    
    # 5. Extent: ratio of contour area to bounding box area
    letter_extent = letter_area / (letter_w * letter_h) if (letter_w * letter_h) > 0 else 0
    
    # 6. Circularity of the letter
    perimeter = cv2.arcLength(parent_letter, True)
    circularity = 4 * np.pi * letter_area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # Draw all holes in blue for comparison
    cv2.rectangle(all_holes_img, (letter_x, letter_y), 
                  (letter_x+letter_w, letter_y+letter_h), (255, 0, 0), 2)
    
    # Classification rules for 'O'
    is_O = (
        0.20 < hole_ratio < 0.60 and          # O has moderate to large hole
        0.25 < hole_relative_y < 0.75 and     # Hole is vertically centered
        0.20 < hole_relative_x < 0.80 and     # Hole is horizontally centered
        0.6 < letter_aspect < 1.4 and         # O is roughly circular
        letter_extent > 0.65 and               # O fills its bounding box well
        circularity > 0.5                      # O is circular
    )
    
    if is_O:
        o_locations.append((letter_x, letter_y, letter_w, letter_h))
        cv2.rectangle(result, (letter_x, letter_y), 
                     (letter_x+letter_w, letter_y+letter_h), (0, 255, 0), 2)
        cv2.putText(result, 'O', (letter_x, letter_y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    else:
        other_holes.append((letter_x, letter_y, letter_w, letter_h))
        cv2.rectangle(all_holes_img, (letter_x, letter_y), 
                     (letter_x+letter_w, letter_y+letter_h), (0, 0, 255), 2)

# Display results
plt.figure(figsize=(18, 10))

plt.subplot(2, 4, 1)
plt.imshow(binary, cmap='gray')
plt.title('1. Binarized Image')
plt.axis('off')

plt.subplot(2, 4, 2)
plt.imshow(filled, cmap='gray')
plt.title('2. Holes Filled\n(Morphological Reconstruction)')
plt.axis('off')

plt.subplot(2, 4, 3)
plt.imshow(holes, cmap='gray')
plt.title('3. Detected Holes\n(All letters with holes)')
plt.axis('off')

plt.subplot(2, 4, 4)
plt.imshow(holes_cleaned, cmap='gray')
plt.title('4. Cleaned Holes\n(After Opening)')
plt.axis('off')

plt.subplot(2, 4, 5)
plt.imshow(cv2.cvtColor(all_holes_img, cv2.COLOR_BGR2RGB))
plt.title(f'5. All Letters with Holes\n({len(hole_contours)} total)')
plt.axis('off')

plt.subplot(2, 4, 6)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title(f'6. Only "O" Detected\n({len(o_locations)} found, {len(other_holes)} filtered out)')
plt.axis('off')

# Create comparison image
comparison = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
for x, y, w, h in o_locations:
    cv2.rectangle(comparison, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(comparison, 'O', (x, y-5), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

plt.subplot(2, 4, 7)
plt.imshow(cv2.cvtColor(comparison, cv2.COLOR_BGR2RGB))
plt.title('7. Final Result:\nOnly "O" marked (Green)')
plt.axis('off')

# Show the filtering criteria used
plt.subplot(2, 4, 8)
plt.axis('off')

plt.tight_layout()
plt.show()

print("="*60)
print(f"Number of 'O' letters detected: {len(o_locations)}")
print(f"Letters with holes filtered out: {len(other_holes)} (p, g, d, etc.)")
print("="*60)
print("\nDetected 'O' locations (x, y, width, height):")
for i, (x, y, w, h) in enumerate(o_locations, 1):
    print(f"  O #{i}: ({x}, {y}, {w}, {h})")