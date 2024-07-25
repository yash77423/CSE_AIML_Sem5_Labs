import numpy as np
import cv2

# define a video capture object. This will return video from the first webcam on your computer.
cap = cv2.VideoCapture(0)

# For saving a recorded video, we create a Video Writer object
# FourCC is a 4-byte code used to specify the video codec. The codecs for Windows is DIVX. FourCC code is passed as cv2.VideoWriter_fourcc(*'XVID') for DIVX.
# Then, the cv2.VideoWriter() function is used as cv2.VideoWriter( filename, fourcc, fps, frameSize )
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,480))

# loop runs if capturing has been initialized.
while(True):
    # reads frames from a camera
    # ret checks return at each frame
    ret, frame = cap.read()

    # Converts to grayscale space, OCV reads colors as BGR
    # frame is converted to gray
    gray = cv2.cvtColor(frame,  cv2.COLOR_BGR2GRAY)

    # output the frame
    out.write(frame)

    # The original input frame is shown in the window
    cv2.imshow('Original', frame)

    # The window showing the operated video stream
    cv2.imshow('frame',gray)

    # Wait for 'a' key to stop the program
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break

# Close the window / Release webcam
cap. release()

# After we release our webcam, we also release the out
out.release()

# De-allocate any associated memory usage
cv2. destroyAllWindows()
