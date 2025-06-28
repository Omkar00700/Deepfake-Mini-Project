import cv2
import numpy as np

# Create a simple test image with skin-like color
skin_img = np.ones((224, 224, 3), dtype=np.uint8)
skin_img[:, :, 0] = 80  # B
skin_img[:, :, 1] = 120  # G
skin_img[:, :, 2] = 160  # R

# Save the test image
cv2.imwrite('skin_test.jpg', skin_img)

print('Skin tone test image created successfully!')
