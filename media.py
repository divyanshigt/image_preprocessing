import cv2
import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
face_mesh=mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5 
)
cap=cv2.VideoCapture(0)
while True:
    flag,frame=cap.read()
    if not flag:
        break
    rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    #process frame to mediapipe model
    results=face_mesh.process(rgb_frame)
    #drawing landmarks
    if results.multi_face_landmarks:
        #loop through detected faces(only 1 here)
        for face_landmarks in results.multi_face_landmarks:
           #draw full face mesh
            mp_drawing.draw_landmarks(
                image=frame,#image to draw
                landmark_list=face_landmarks, #detected face landmarks
                connections=mp_face_mesh.FACEMESH_TESSELATION, #mesh structure to connect landmarks

            )
    cv2.imshow("MediaPipe Face Mesh",frame)
    key=cv2.waitKey(1)& 0xFF
    if key==27: #ESC key to exit
        break
cap.release()
cv2.destroyAllWindows()
