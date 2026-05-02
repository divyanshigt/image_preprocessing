import cv2
import numpy as np
import mediapipe as mp
mp_face_mesh=mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
static_image_mode=False,
max_num_faces=2,
refine_landmarks=True,
min_detection_confidence=0.5,
min_tracking_confidence=0.5 
)
glasses=cv2.imread("be.png",cv2.IMREAD_UNCHANGED)
earring=cv2.imread("ear.png",cv2.IMREAD_UNCHANGED)
if glasses is None:
 raise FileNotFoundError("be.png not loaded")
if earring is None:
 raise FileNotFoundError("ear.png not loaded")
if glasses.shape[2]==3:
 glasses=cv2.cvtColor(glasses,cv2.COLOR_BGR2BGRA)
if earring.shape[2]==3:
 earring=cv2.cvtColor(earring,cv2.COLOR_BGR2BGRA)
cap=cv2.VideoCapture(0)
while True:
 flag,frame=cap.read()
 if not flag:
  break
 rgb_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
 results=face_mesh.process(rgb_frame)
 if results.multi_face_landmarks:
  for face_landmarks in results.multi_face_landmarks:
   h,w,_=frame.shape

   #eyes for glasses
   x1=int(face_landmarks.landmark[33].x*w)
   y1=int(face_landmarks.landmark[33].y*h)
   x2=int(face_landmarks.landmark[263].x*w)
   y2=int(face_landmarks.landmark[263].y*h)
   face_width=int(np.hypot(x2-x1,y2-y1))

   #glasses
   overlay=glasses
   ow=face_width+65
   oh=int(ow*0.50)
   resized=cv2.resize(overlay,(ow,oh))
   xc=(x1+x2)//2
   yc=(y1+y2)//2
   xo=int(xc-ow/2)
   yo=int(yc-oh/2+5)
   if not(xo<0 or yo<0 or xo+ow>w or yo+oh>h):
    img=resized[:,:,:3]
    mask=resized[:,:,3]
    roi=frame[yo:yo+oh,xo:xo+ow]
    bg=cv2.bitwise_and(roi,roi,mask=cv2.bitwise_not(mask))
    fg=cv2.bitwise_and(img,img,mask=mask)
    frame[yo:yo+oh,xo:xo+ow]=cv2.add(bg,fg)

   #earrings (left & right)
   left_ear=face_landmarks.landmark[234]
   right_ear=face_landmarks.landmark[454]

   for idx,ear in enumerate([left_ear,right_ear]):
    ex=int(ear.x*w)
    ey=int(ear.y*h)
    ew=int(face_width*0.70)
    eh=int(ew*1.15)
    resized=cv2.resize(earring,(ew,eh))
    if idx==1:
     resized=cv2.flip(resized,1)
    if idx==0:
     xo=int(ex-ew/2-5)
    else:
     xo=int(ex-ew/2+5)
    yo=int(ey+10)
    if not(xo<0 or yo<0 or xo+ew>w or yo+eh>h):
     img=resized[:,:,:3]
     mask=resized[:,:,3]
     roi=frame[yo:yo+eh,xo:xo+ew]
     bg=cv2.bitwise_and(roi,roi,mask=cv2.bitwise_not(mask))
     fg=cv2.bitwise_and(img,img,mask=mask)
     frame[yo:yo+eh,xo:xo+ew]=cv2.add(bg,fg)

   #lipstick (only lips, not teeth)
   upper_lip=[61,185,40,39,37,0,267,269,270,409,291,308,415,310,311,312,13,82,81,80,191,78]
   lower_lip=[61,146,91,181,84,17,314,405,321,375,291,308,324,318,402,317,14,87,178,88,95,78]
   upper_points=[]
   lower_points=[]
   for id in upper_lip:
    x=int(face_landmarks.landmark[id].x*w)
    y=int(face_landmarks.landmark[id].y*h)
    upper_points.append([x,y])
   for id in lower_lip:
    x=int(face_landmarks.landmark[id].x*w)
    y=int(face_landmarks.landmark[id].y*h)
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
   lip_color[:]=(100,0,100)
   alpha=0.65
   colored=cv2.addWeighted(frame,1-alpha,lip_color,alpha,0)
   frame[mask>20]=colored[mask>20]
        #beauty filter (smooth + glow + thoda fair)
#fairness filter (no blur, clear face)
   lab=cv2.cvtColor(frame,cv2.COLOR_BGR2LAB)

   l,a,b=cv2.split(lab)

    #increase brightness (main fairness)
   l=cv2.add(l,25)

    #slight warm tone (natural fairness)
   b=cv2.add(b,5)

   lab=cv2.merge([l,a,b])
   frame=cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
 cv2.imshow("Face Filter",frame)
 if cv2.waitKey(1)&0xFF==27:
  break
cap.release()
cv2.destroyAllWindows()
