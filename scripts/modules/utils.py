import cv2

def show_image(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 2000, 1500)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
