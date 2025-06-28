import cv2
import numpy as np

# Create a simple test image (gray square)
test_img = np.ones((224, 224, 3), dtype=np.uint8) * 128

# Save the test image
cv2.imwrite('test_image.jpg', test_img)
print('Test image created successfully!')
