import cv2
import mediapipe as mp


# TODO: Landmark coordinates 
POS_EYE_R = [] 
POS_EYE_L = [] 
POS_MOUTH = []


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
            
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = face_mesh.process(image)

            if results.multi_face_landmarks:
                # landmark
                landmark_list = []
                # normalized coordinates, 
                # ref: stackoverflow.com/questions/67141844/how-do-i-get-the-coordinates-of-face-mash-landmarks-in-mediapipe
                for fp in results.multi_face_landmarks[0].landmark:
                    landmark_list.append([fp.x, fp.y, fp.z])
                
                # denormalize
                for i in range(len(landmark_list)):
                    landmark_list[i][0] *= image.shape[1]
                    landmark_list[i][1] *= image.shape[0]
                
                # Draw the face mesh annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        mp_drawing.draw_landmarks(
                            image=image,
                            landmark_list=face_landmarks,
                            connections=mp_face_mesh.FACE_CONNECTIONS,
                            landmark_drawing_spec=drawing_spec,
                            connection_drawing_spec=drawing_spec)
                
                for i in range(len(landmark_list)):
                    pos = (int(landmark_list[i][0]), int(landmark_list[i][1]))
                    cv2.putText(image, f'{i}', pos, cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255,255,255), 1, cv2.LINE_AA)

            cv2.imshow('Landmark coordinates test', image)
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break
        
        cv2.destroyAllWindows()
        cap.release()
