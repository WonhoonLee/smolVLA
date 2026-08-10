import cv2

cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

print("camera 0 open:", cap0.isOpened())
print("camera 1 open:", cap1.isOpened())

ret0, frame0 = cap0.read()
ret1, frame1 = cap1.read()

print("camera 0 read:", ret0)
print("camera 1 read:", ret1)

if ret0 and ret1:
    frame0 = cv2.resize(frame0, (640, 480))
    frame1 = cv2.resize(frame1, (640, 480))

    combined = cv2.hconcat([frame0, frame1])

    cv2.imwrite("camera_compare.jpg", combined)
    cv2.imwrite("camera_0_test.jpg", frame0)
    cv2.imwrite("camera_1_test.jpg", frame1)

    print("saved: camera_compare.jpg")
    print("saved: camera_0_test.jpg")
    print("saved: camera_1_test.jpg")

cap0.release()
cap1.release()