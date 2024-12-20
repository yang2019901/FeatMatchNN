## use orb/sift to match features between two images
import numpy as np
import cv2


## use ransac to find homography
def find_homography(kp1, kp2, matches):
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    return M, mask


def match_features(img1, img2, mask1=None, mask2=None):
    detector = cv2.ORB_create()
    detector: cv2.ORB
    # find the keypoints and descriptors with detector
    kp1, des1 = detector.detectAndCompute(img1, mask1)
    kp2, des2 = detector.detectAndCompute(img2, mask2)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Match descriptors
    matches = bf.match(des1, des2)
    # Sort them in the order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    # find homography
    M, mask = find_homography(kp1, kp2, matches)
    print(M)
    # filter out the outliers
    matchesMask = mask.ravel().tolist()
    # draw the matches
    img3 = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None, matchesMask=matchesMask, flags=2
    )
    # show the result
    cv2.imshow("Matches", img3)
    cv2.waitKey(0)


def main():
    img1 = cv2.imread("img1.jpg", 0)
    img2 = cv2.imread("img2.jpg", 0)

    img1 = cv2.resize(img1, (480, 640))
    img2 = cv2.resize(img2, (480, 640))

    # mask1 = np.zeros_like(img1)
    # mask2 = np.zeros_like(img2)
    # mask1[120:430, 120:420] = 255
    # mask2[200:470, 80:300] = 255
    # cv2.imshow("masked img1", cv2.bitwise_and(img1, img1, mask=mask1))
    # cv2.imshow("masked img2", cv2.bitwise_and(img2, img2, mask=mask2))
    # match_features(img2, img1, mask2, mask1)
    match_features(img1, img2)


if __name__ == "__main__":
    main()
