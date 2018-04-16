#!/usr/bin/python

import cv2

cap = cv2.VideoCapture(0)

number = 0
set_dir   = 'AED500'
name_type = 'AED500_front'

print ("Photo capture enabled! Press esc to take photos!")

while True:
    ret, frame = cap.read()
    cv2.imshow('Color Picture', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        fname = 'Data/' + set_dir + '/' + name_type + "_" + str(number) + ".png"
        #gray   = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #resize = cv2.resize(frame,(360,360), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite(fname , frame)
        print ("Saving image : "+ str(fname))
        print (frame.shape)
        number+=1

    if number == 100:
        break

cap.release()
cv2.destroyAllWindows()
