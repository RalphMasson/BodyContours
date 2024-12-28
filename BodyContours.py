import cv2
from handTracker import *
import mediapipe as mp
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation
mp_pose = mp.solutions.pose

pose_img = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5, model_complexity=1)
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5,
                          min_tracking_confidence=0.5, model_complexity=1)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

def body_contour(path_file):
    IMAGE_FILES = [path_file]
    BG_COLOR = (0, 0, 0)  # black
    MASK_COLOR = (255, 255, 255)  # white

    with mp_selfie_segmentation.SelfieSegmentation(
            model_selection=0) as selfie_segmentation:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            results = selfie_segmentation.process(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.4
            fg_image = np.zeros(image.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            output_image = np.where(condition, fg_image, bg_image)
            gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            image_copy = image.copy()
            image_copy = np.zeros(image.shape,dtype=np.uint8)
            output_img = image.copy()
            RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            results = pose_img.process(RGB_img)
            RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(RGB_img)
            landmarks_c=(234,63,247)
            connection_c=(117,249,77)
            thickness=10
            circle_r=8
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(image_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                        mp_drawing.DrawingSpec(connection_c, thickness, circle_r))
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x_pixel = int(landmark.x * image_width)
                        y_pixel = int(landmark.y * image_height)
                        cv2.circle(image_copy, (x_pixel, y_pixel), 2, (0, 255, 0), -1)

            cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
            cv2.namedWindow('win_name', cv2.WINDOW_NORMAL)
            cv2.moveWindow('win_name', 0, 0)
            cv2.imshow('win_name', image_copy)
            cv2.resizeWindow('win_name', image_width//3, image_height//3)
            cv2.waitKey(0)


def body_contour_video(path_video):
    BG_COLOR = (0, 0, 0)  # black
    MASK_COLOR = (255, 255, 255)  # white

    cap = cv2.VideoCapture(path_video)
    # cap = cv2.VideoCapture('rtsp://admin:admin@192.168.1.24:1935')
    if not cap.isOpened():
        print(f"Erreur : Impossible d'ouvrir la vidéo {path_video}")
        return

    with mp_selfie_segmentation.SelfieSegmentation(model_selection=0) as selfie_segmentation:
        frame_counter = 0
        frame_skip = 2  # Traiter une image sur deux
        while cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            if not ret:
                break

            if frame_counter % frame_skip != 0:
                frame_counter += 1
                continue

            frame_counter += 1
            detector = HandTracker(detectionCon=1)
            image_height, image_width, _ = frame.shape
            RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            results = selfie_segmentation.process(RGB_img)
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.4
            fg_image = np.zeros(frame.shape, dtype=np.uint8)
            fg_image[:] = MASK_COLOR
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            output_image = np.where(condition, fg_image, bg_image)
            gray = cv2.cvtColor(output_image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            frame_copy = frame.copy()
            detector.findHands(frame_copy)

            frame_copy = np.zeros(frame.shape,dtype=np.uint8)
            positions = detector.getPostion(frame_copy, draw=True)
            upFingers = detector.getUpFingers(frame_copy)
            output_img = frame.copy()
            results = pose_img.process(RGB_img)
            # RGB_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)
            results_face = face_mesh.process(RGB_img)
            landmarks_c=(234,63,247)
            connection_c=(117,249,77)
            thickness=10
            circle_r=8
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame_copy, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(landmarks_c, thickness, circle_r),
                                        mp_drawing.DrawingSpec(connection_c, thickness, circle_r))
            if results_face.multi_face_landmarks:
                for face_landmarks in results_face.multi_face_landmarks:
                    for idx, landmark in enumerate(face_landmarks.landmark):
                        x_pixel = int(landmark.x * image_width)
                        y_pixel = int(landmark.y * image_height)
                        cv2.circle(frame_copy, (x_pixel, y_pixel), 2, (0, 255, 0), -1)
            cv2.drawContours(frame_copy, contours, -1, (0, 255, 0), 2)
            cv2.imshow('Body Contour Video', frame_copy)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
