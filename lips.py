import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh
mp_drawing=mp.solutions.drawing_utils
face_mesh=mp_face_mesh.FaceMesh(
static_image_mode=False,
max_num_faces=2,
refine_landmarks=True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5 
)
glasses=cv2.imread("sunglasses.webp",cv2.IMREAD_UNCHANGED) #read glasses image with alpha channel
if glasses is None:
 raise FileNotFoundError("sunglasses.webp not loaded")
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
   h,w,_=frame.shape
   #left eye=33
   #right eye=263
   left_eye_point=face_landmarks.landmark[33]
   right_eye_point=face_landmarks.landmark[263]
   x1,y1=int(left_eye_point.x*w),int(left_eye_point.y*h)
   x2,y2=int(right_eye_point.x*w),int(right_eye_point.y*h)
   #calculate glasses size
   face_width=int(np.hypot(x2-x1,y2-y1))

   #glasses overlay
   overlay=glasses
   overlay_width=face_width+40
   overlay_height=int(overlay_width*0.4)
   resized_overlay=cv2.resize(overlay,(overlay_width,overlay_height))
   x_center=(x1+x2)//2
   y_center=(y1+y2)//2
   #top left corner of glasses 
   x_offset=int(x_center-overlay_width/2)
   y_offset=int(y_center-overlay_height/2)
   if not(x_offset<0 or y_offset<0 or x_offset+overlay_width>w or y_offset+overlay_height>h):
    #split channel
    overlay_img=resized_overlay[:,:,:3]
    mask_overlay=resized_overlay[:,:,3] #alpha channel as mask
    mask_inv=cv2.bitwise_not(mask_overlay)
    roi=frame[y_offset:y_offset+overlay_height,x_offset:x_offset+overlay_width] #region of interest where glasses will be placed
    bg=cv2.bitwise_and(roi,roi,mask=mask_inv) #background where glasses will be placed
    fg=cv2.bitwise_and(overlay_img,overlay_img,mask=mask_overlay) #glasses foreground
    combined=cv2.add(bg,fg) #combine background and foreground
    frame[y_offset:y_offset+overlay_height,x_offset:x_offset+overlay_width]=combined #place combined image back to frame

   #lipstick
    upper_lip=[61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
    lower_lip=[61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78]
    upper_points=[]
    lower_points=[]
    for id in upper_lip:
        landmark=face_landmarks.landmark[id]
        x=int(landmark.x*w)
        y=int(landmark.y*h)
        upper_points.append([x,y])
    for id in lower_lip:
        landmark=face_landmarks.landmark[id]
        x=int(landmark.x*w)
        y=int(landmark.y*h)
        lower_points.append([x,y])
    upper_points=np.array(upper_points,np.int32)
    lower_points=np.array(lower_points,np.int32)
    mask=np.zeros((h,w),dtype=np.uint8)
    cv2.fillPoly(mask,[upper_points],255)
    cv2.fillPoly(mask,[lower_points],255)
    kernel=np.ones((3,3),np.uint8)
    mask=cv2.erode(mask,kernel,iterations=1)
    mask=cv2.GaussianBlur(mask,(7,7),0)
    lip_color=np.zeros_like(frame)
    lip_color[:]=(70,80,160)
    alpha=0.65
    colored=cv2.addWeighted(frame,1-alpha,lip_color,alpha,0)
    frame[mask>20]=colored[mask>20]

 cv2.imshow("MediaPipe Face Mesh",frame)
 key=cv2.waitKey(1)&0xFF
 if key==27: #ESC key to exit
  break
cap.release()
cv2.destroyAllWindows()