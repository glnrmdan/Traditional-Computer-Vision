import cv2
import numpy as np

# load images
chicken1 = cv2.imread('img/chicken 1.jpg')
chicken2 = cv2.imread('img/chicken 2.jpg')

# ORB Detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB
keypoints1, descriptors1 = orb.detectAndCompute(chicken1, None)
keypoints2, descriptors2 = orb.detectAndCompute(chicken2, None)

# Brute-Force Matching with Hamming distance
brute = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors
matches = brute.match(descriptors1, descriptors2)

# Sort the matches based on distance
matches = sorted(matches, key=lambda x: x.distance)

# Draw the matches
N = 1000
result = cv2.drawMatches(chicken1, keypoints1, chicken2, keypoints2, matches[:N], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Display the matches
cv2.imshow('ORB Matches', result)
cv2.waitKey(0)
cv2.destroyAllWindows()