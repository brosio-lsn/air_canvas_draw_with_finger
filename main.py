from collections import deque
import cv2 as cv
import numpy as np


# function to draw the initial rectangles used to colect pixels of the hand to perform histogram analysis
def draw_rectangles(image):
    copy = image.copy()
    # mask to isolate only the parts of the image inside the rectangles
    rectangles_mask = np.zeros(image.shape[:2], dtype='uint8')

    # first option of rectangles:
    cv.rectangle(copy, (int(len(image[0]) / 4) + 100, len(image)),
                 (int(len(image[0]) / 4 * 3) - 100, len(image) - int(len(image[0]) / 3)), (0, 255, 0), 2)
    # turn to white the pixels inside the rectangle in our mask
    for x in range(int(len(image[0]) / 4) + 100, int(len(image[0]) / 4 * 3) - 100):
        for y in range(len(image) - 1, len(image) - int(len(image[0]) / 3), -1):
            rectangles_mask[y][x] = 255

    cv.rectangle(copy, (int(len(image[0]) / 4 * 3) - 75, len(image) - int(len(image[0]) / 3)),
                 (int(len(image[0]) / 4 * 3) - 50, 0), (0, 255, 0), 2)
    # turn to white the pixels inside the rectangle in our mask
    for x in range(int(len(image[0]) / 4 * 3) - 75, int(len(image[0]) / 4 * 3) - 50):
        for y in range(len(image) - int(len(image[0]) / 3), 0, -1):
            rectangles_mask[y][x] = 255

    #  second option of rectangles:
    # # x coordinates of the 3 columns where the 6 rectangles will be displayed
    # xs = np.array([int(len(image[0]) / 2) - 85, int(len(image[0]) / 2) - 15, int(len(image[0]) / 2) + 55])
    # # y coordinates of the 3 rows where the 6 rectangles will be displayed
    # ys = np.array([int(len(image) / 2) - 115, int(len(image) / 2) + 15, int(len(image) / 2) + 155])
    # width = 30
    # height = 30
    # for x in xs:
    #     for y in ys:
    #         # fill pixel_values_in_rectangles
    #         for i in range(y, y + height):
    #             for j in range(x, x + width):
    #                 rectangles_mask[i][j] = 255
    #         # drawing the rectangles
    #         cv.rectangle(copy, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return copy, rectangles_mask


# calculate the hand histogram given the image and the rectangles mask
def caclculate_hist(image, rectangles_mask):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    hist = cv.calcHist([hsv], [0, 1], rectangles_mask, [180, 256], [0, 180, 0, 256])
    return cv.normalize(hist, hist, 155, 255, cv.NORM_MINMAX)


# function to isolate the hand from the given image, using a mask created from the given histogram
def isolate_hand(image, hand_hist):
    # creating the initial mask to isolate the hand with a mask based on its coloration histogram created above
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    dst = cv.calcBackProject([hsv], [0, 1], hand_hist, [0, 180, 0, 256], 1)
    ret, thresh = cv.threshold(dst, 155, 255, cv.THRESH_BINARY)

    # morphological transformations to improve the mask
    kernelOpening = np.ones((8, 8), np.uint8)
    kernelClosing = np.ones((31, 31), np.uint8)
    opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernelOpening)
    morpho = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernelClosing)
    disc = cv.getStructuringElement(cv.MORPH_ELLIPSE, (61, 61))
    cv.filter2D(morpho, -1, disc, morpho)

    # applying the mask to the image
    return cv.bitwise_and(image, image, mask=morpho)


# function to find the contour of the hand given the image in which the hand has been isolated with a mask
def find_hand_contours(masked_image):
    # turning image to grayscale before finding the contours
    gray = cv.cvtColor(masked_image, cv.COLOR_BGR2GRAY)
    contours, hierarchy = cv.findContours(gray, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
    # finding the maximum contour which is very likely the contour of the hand
    if (len(contours) == 0):
        print("contours is empty")
        return []
    else:
        max_contour = contours[0]
        for contour in contours:
            if (cv.contourArea(contour) > cv.contourArea(max_contour)):
                max_contour = contour
    # returning the hand contour
    return max_contour


# function to find the centroid of the hand contour
def find_contour_centroid(contour):
    moments = cv.moments(contour)
    # add 1e-5 to avoid division by zero
    return (moments['m10'] / (moments['m00'] + 1e-5), moments['m01'] / (moments['m00'] + 1e-5))


# function to calculate square distance between 2 points
def square_distance(x1, y1, x2, y2):
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2)


# function to find the furthest convexity defects in the contour
def find_furthest_convexity_defect(contour, centroid):
    # find the hull of the hand contour
    hull = cv.convexHull(contour, returnPoints=False)
    # find the convexity defects
    convexity_defects = cv.convexityDefects(contour, hull)
    # find the convexity defect with the maximum distance from the centroid
    if (convexity_defects is None):
        print("no convexity defects")
        return None
    else:
        furthest = convexity_defects[0][0][2]
        for defect in convexity_defects:
            start, end, furthest2, distance = defect.reshape(4)
            d1 = square_distance(centroid[0], centroid[1], contour[furthest2][0][0], contour[furthest2][0][1])
            d2 = square_distance(centroid[0], centroid[1], contour[furthest][0][0], contour[furthest][0][1])
            if (d1 > d2):
                furthest = furthest2
    return tuple(contour[furthest][0])


# function to draw all the points in the given deque
def draw_points(image, d):
    for point in d:
        print(len(d))
        cv.circle(image, point, 5, (255, 0, 0), -1)


image = cv.imread('hand_frame.jpg')
copy = image.copy()

# result, rectangles_mask = draw_rectangles(image)
# masked = isolate_hand(image, rectangles_mask)
# hand_contour = find_hand_contours(masked)
# centroid=find_contour_centroid(hand_contour)
# cv.circle(find_furthest_convexity_defect(hand_contour, copy, centroid), (int(centroid[0]), int(centroid[1])), 6, (0,0,255), -1)
# cv.imshow('Video', copy)
# cv.waitKey(0)


# test for a video
capture = cv.VideoCapture(0)
time = 0
# this list will store the previous positions of the tip of the finger so that we can display some of the past position
# (makes a more realistic canvas)
dq = deque()
time_for_hand_placing = 2000
while (capture.isOpened()):
    isTrue, frame = capture.read()
    # the user has 3 seconds to place his hands nicely according to the rectangles
    rectangles, rectangles_mask = draw_rectangles(frame)
    if (time < time_for_hand_placing):
        cv.putText(rectangles, str((time_for_hand_placing - time)), (10, 25), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                   1, 2)
        cv.imshow('Video', rectangles)
    if (time == time_for_hand_placing):
        # after 3 seconds, the hand histogram is calulated
        hist = caclculate_hist(frame, rectangles_mask)
    if (time > time_for_hand_placing):
        # when time is superior to 3 seconds, the tip of the finger is marked on the screen
        masked = isolate_hand(frame, hist)
        hand_contour = find_hand_contours(masked)
        if (len(hand_contour) != 0):
            centroid = find_contour_centroid(hand_contour)
            furthest_convexity_defect = find_furthest_convexity_defect(hand_contour, centroid)
            if (not (furthest_convexity_defect is None)):
                dq.appendleft(furthest_convexity_defect)
                draw_points(frame, dq)
                if (time > 4000):
                    # if time is superior to 4 seconds, we remove the point in the dequeue that was first added
                    # (otherwise, too many points would appear)
                    dq.pop()
                # we deaw the centroid
                cv.circle(frame, (int(centroid[0]), int(centroid[1])), 6, (0, 0, 255), -1)
        cv.imshow('Video', frame)
        cv.imshow("ma", masked)
    cv.waitKey(10)
    time += 10
    print(time)
    # below helps allows to turn of the camera by pressing q
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
capture.release()
cv.destroyAllWindows()
