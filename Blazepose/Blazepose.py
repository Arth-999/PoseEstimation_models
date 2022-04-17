import cv2
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def body_measurements(x1,x2):
    lx=results.pose_landmarks.landmark[x1].x 
    rx=results.pose_landmarks.landmark[x2].x 
    ly=results.pose_landmarks.landmark[x1].y 
    ry=results.pose_landmarks.landmark[x2].y 
    lz=results.pose_landmarks.landmark[x1].z 
    rz=results.pose_landmarks.landmark[x2].z 
    dist=((lx-rx)**2+(ly-ry)**2+(lz-rz)**2)**(1/2)
    print(dist)
    print(dist*image_width)

# For static images:
IMAGE_FILES = [r"D:\capstone\working project\blazepose\arth2.jpeg"]
BG_COLOR = (192, 192, 192)  # gray
with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
    for idx, file in enumerate(IMAGE_FILES):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        print(image_height, image_width)
        if not results.pose_landmarks:
            continue
        print('shoulder width:')
        body_measurements(mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.RIGHT_SHOULDER)
        print('waist:')
        body_measurements(mp_pose.PoseLandmark.LEFT_HIP,mp_pose.PoseLandmark.RIGHT_HIP)
        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
        # condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        # bg_image = np.zeros(image.shape, dtype=np.uint8)
        # bg_image[:] = BG_COLOR
        # annotated_image = np.where(condition, annotated_image, bg_image)
        
        # Draw pose landmarks on the image.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imwrite(r'D:\capstone\working project\blazepose\pose.annotated_image' +
                    str(idx) + '.png', annotated_image)
        
        # Plot pose world landmarks.
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
