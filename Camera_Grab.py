#!/usr/bin/python

import cv2

cap = cv2.VideoCapture(0)

number = 0
set_dir   = 'AED50'
name_type = 'AED50_back'

print ("Photo capture enabled! Press esc to take photos!")

while True:
    ret, frame = cap.read()
    cv2.imshow('Color Picture', frame)

    if cv2.waitKey(1) & 0xFF == 27:
        #gray   = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        #resize = cv2.resize(frame,(360,360), interpolation = cv2.INTER_NEAREST)
        cv2.imwrite('Data/' + set_dir + '/' + name_type + "_" + str(number) + ".png", frame)
        print ("Saving image number: " + str(number))
	print (frame.shape)
        number+=1

    if number == 100:
        break

cap.release()
cv2.destroyAllWindows()
