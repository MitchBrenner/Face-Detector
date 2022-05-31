import cv2
from random import randrange

# This will make a classifier which is a detector -> it will classify it as a face
# the argument is the training data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# image of face to detect
img = cv2.imread('ronaldo_image.jpeg')


# need to make the image black and white because the algo only takes black and white
# greyscale
# its BGR not RGB in cv
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
# whatever this classifier is then we want to detect all the faces with multiscale (size doesn't matter)
# this is based off whatever you chose to detect, and we chose faces
# this returns a tuple with the otp left coordinate and the width and height (x, y, w, h)
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

# make rectangle over coordinates
for (x, y, w, h) in face_coordinates:
    cv2.rectangle(img, (x, y), (x + w, y + h), (randrange(128, 256), randrange(128, 256), randrange(128, 256)), 5)

print(face_coordinates)

# this pops up a window with the image
cv2.imshow('Face Detector', img)

# pauses the execution of your code, so this will wait until you press any key
# allows image to see stay for longer than a split second
# this is just how opencv is
cv2.waitKey()

print("Code is complete")
