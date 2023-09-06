import cv2
from .SIFT import SIFT

class BFMatcher:
    def __init__(self):
        self.bf = cv2.BFMatcher()
        # self.bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)


    def run(self, query_img, train_img, query_keypoints, train_keypoints, query_descriptor, train_descriptor):
        matches = self.bf.knnMatch(query_descriptor, train_descriptor,k=2)
        # matches = self.bf.match(query_descriptor, train_descriptor)

        # ratio test 적용
        good = []
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                good.append([m])

        result = cv2.drawMatchesKnn(query_img, query_keypoints, train_img,train_keypoints,good,None,flags=2)
        return result

if __name__ == "__main__":
    query_img = cv2.imread('/home/yoonk/pipline/images/oxford.jpg', cv2.IMREAD_COLOR)
    train_img = cv2.imread('/home/yoonk/pipline/images/oxford2.jpg', cv2.IMREAD_COLOR)

    sift = SIFT()
    query_output = sift.run(query_img)
    train_output = sift.run(train_img)

    query_keypoints, query_descriptor = sift.sift.detectAndCompute(query_img,None)
    train_keypoints, train_descriptor = sift.sift.detectAndCompute(train_img,None)

    bfMatcher = BFMatcher()
    result_sift = bfMatcher.run(query_keypoints, train_keypoints, query_descriptor, train_descriptor)

    window_name = "BF with SIFT"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 2000, 1500)
    cv2.imshow(window_name, result_sift)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#######

# orb = cv2.ORB_create()

# # BFMatcher 객체 생성
# bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

# # 디스크립터들 매칭시키기
# matches = bf.match(des1,des2)

# # 거리에 기반하여 순서 정렬하기
# matches = sorted(matches, key = lambda x:x.distance)

# # 첫 10개 매칭만 그리기
# # flags=2는 일치되는 특성 포인트만 화면에 표시!
# res = cv2.drawMatches(qimg,kp1,timg,kp2,matches[:10],res1,flags=2)

# winname = "Feature Matching"
# cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
# cv2.resizeWindow(winname, 2000, 1500)
# cv2.imshow(winname ,res)
# cv2.waitKey(0)
# cv2.destroyAllWindows()