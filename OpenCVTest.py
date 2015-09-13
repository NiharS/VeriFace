import cv2
import numpy as np
import numpy.linalg as la
from time import sleep

squeeze = 60
cap = cv2.VideoCapture(0)
i=0
imgs = []
xs = []
ys = []
ws = []
hs = []
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
raw_input("press enter")
while(i < 10):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("frame", gray)
    faces = face_cascade.detectMultiScale(gray, 2, 3)
    if (len(faces) > 0):
        x,y,w,h = faces[0]
        face = gray
        xs.append(x)
        ys.append(y)
        ws.append(w)
        hs.append(h)
        print x, y, w, h
        #face = gray[x:x+w,y:y+h]
        cv2.imwrite("face_"+str(i)+".png", face)
        imgs.append(face)
        print "lengths:",len(imgs), len(xs)
        i+=1
    key = cv2.waitKey(100)
    if (key & 0xFF == ord('q')): break
cap.release()
print "done with cap"
#for i in range(10):
#    cv2.imshow("image", imgs[i])
#    cv2.waitKey(1000)
w_avg = int(sum(ws) * 1.0 / len(ws))
h_avg = int(sum(hs) * 1.0 / len(hs))
face_crops = []
tmp = -70
for i in range(len(imgs)): #for (int i = 0; i < imgs.length; i++)
    startx = xs[i] + tmp
    starty = ys[i] - tmp
    face_crops.append(imgs[i][startx + squeeze : startx + w_avg, starty : starty + h_avg - squeeze])
    print("shape:", face_crops[-1].shape)
k=0
vecs = []
for i in face_crops:
    cv2.imwrite("facecrop_" + str(k) + ".png", i)
    k+=1
    print "old i shape:", i.shape
    i=i.reshape(-1)
    print "i shape:",i.shape
    smaller_vec = []
    for j in range(0, len(i)-5, 5):
        smaller_vec.append((int(i[j]) + int(i[j+1]) + int(i[j+2]) + int(i[j+3]) + int(i[j+4]))/5.)
    smaller_vec = np.array(smaller_vec)
    vecs.append(smaller_vec)
vecs = np.array(vecs)
print "vecs shape:", vecs.shape
print "vecs:\n",vecs
f_avg = np.sum(vecs, axis=0) * 1.0 / len(vecs)

print "average of vecs:", f_avg.shape, '\n', f_avg

A = np.array([i-f_avg for i in vecs])

u, s, vt = la.svd(A.T)

for z in range(5,0,-1):
    print z
    sleep(1)

epsilons = []

x_n = []

for i in range(len(vecs)):
    x_n.append(u.T.dot(A[i]))
    
x_n = np.array(x_n)

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
input_face_loc = face_cascade.detectMultiScale(gray, 2, 3)
while(len(input_face_loc) == 0):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_face_loc = face_cascade.detectMultiScale(gray, 2, 3)
x,y,w,h = input_face_loc[0]
startx = x + tmp;
starty = y - tmp;
input_face = gray[startx : startx + w, starty : starty + h]
dsize = (w_avg, h_avg)
face = cv2.resize(input_face, dsize)
face = face[squeeze:, :-squeeze]
cv2.imwrite("finalface.png", face)

flattened = face.reshape(-1)
smaller_face = []
for i in range(0, len(flattened)-5, 5):
    smaller_face.append((int(flattened[i]) + int(flattened[i+1]) + int(flattened[i+2]) + int(flattened[i+3]) + int(flattened[i+4]))/5.)
smaller_face = np.array(smaller_face)
    
x = u.T.dot(smaller_face-f_avg)

norms = []

for i in x_n:
    norms.append(la.norm(x, i.any()))
print "Likeness=",(100-min(norms)/10000)

cv2.destroyAllWindows()