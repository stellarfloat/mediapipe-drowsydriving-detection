import cv2 
import mediapipe as mp
import math


# Landmark coordinates 
POS_MOUTH = ((12, 14), (78, 306)) # ((vertical), (horizontal))
DETECTION_THRESHOLD = 0.45

def feature_mouth(landmark_list):
    V = math.dist(landmark_list[POS_MOUTH[0][0]], landmark_list[POS_MOUTH[0][1]])
    H = math.dist(landmark_list[POS_MOUTH[1][0]], landmark_list[POS_MOUTH[1][1]])
    return V / H


if __name__ == '__main__':
    mp_drawing = mp.solutions.drawing_utils
    mp_face_mesh = mp.solutions.face_mesh

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
    min_detection_confidence=0.35,
    min_tracking_confidence=0.65) as face_mesh:

        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            # To improve performance, optionally mark the image as not writeable to pass by reference.
            image.flags.writeable = False
            results = face_mesh.process(image)

            image.flags.writeable = True
            #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_face_landmarks:
                landmark_list = []
                # normalized coordinates, 
                # ref: stackoverflow.com/questions/67141844/how-do-i-get-the-coordinates-of-face-mash-landmarks-in-mediapipe
                for fp in results.multi_face_landmarks[0].landmark:
                    landmark_list.append([fp.x, fp.y, fp.z])
                
                # denormalize
                for i in range(len(landmark_list)):
                    landmark_list[i][0] *= image.shape[1]
                    landmark_list[i][1] *= image.shape[0]

                # Draw the face mesh annotations on the image
                for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                
                # Draw feature_mouth
                pt1 = tuple(map(int, landmark_list[POS_MOUTH[0][0]][:2]))
                pt2 = tuple(map(int, landmark_list[POS_MOUTH[0][1]][:2]))
                cv2.line(image, pt1, pt2, 
                    (0, 0, 255), 1, cv2.LINE_AA)
                pt1 = tuple(map(int, landmark_list[POS_MOUTH[1][0]][:2]))
                pt2 = tuple(map(int, landmark_list[POS_MOUTH[1][1]][:2]))
                cv2.line(image, pt1, pt2, 
                    (0, 0, 255), 1, cv2.LINE_AA)
                
                ft_mouth = feature_mouth(landmark_list)

            else:
                ft_mouth = -1
            
            
            cv2.putText(image, f'ft_mouth: {ft_mouth:.2f}', (int(0.03*image.shape[1]), int(0.06*image.shape[0])),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


            cv2.imshow('Drowsy driving detection', image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()
        cap.release()
