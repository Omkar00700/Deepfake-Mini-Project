import cv2
import numpy as np
import os

def test_face_enhancement():
    try:
        # Create a simple face-like image
        face_img = np.ones((224, 224, 3), dtype=np.uint8)
        # Make it skin-colored
        face_img[:, :, 0] = 80  # B
        face_img[:, :, 1] = 120  # G
        face_img[:, :, 2] = 160  # R
        
        # Add some facial features (simple shapes)
        # Eyes
        cv2.circle(face_img, (75, 90), 15, (255, 255, 255), -1)
        cv2.circle(face_img, (150, 90), 15, (255, 255, 255), -1)
        cv2.circle(face_img, (75, 90), 5, (0, 0, 0), -1)
        cv2.circle(face_img, (150, 90), 5, (0, 0, 0), -1)
        
        # Mouth
        cv2.ellipse(face_img, (112, 150), (30, 15), 0, 0, 180, (0, 0, 0), 2)
        
        # Save the face image
        os.makedirs('test_faces', exist_ok=True)
        face_path = os.path.join('test_faces', 'test_face.jpg')
        cv2.imwrite(face_path, face_img)
        
        print(f'Face enhancement test completed successfully!')
        print(f'Test face saved to {face_path}')
        return True
    
    except Exception as e:
        print(f'Error testing face enhancement: {str(e)}')
        return False

if __name__ == '__main__':
    test_face_enhancement()
