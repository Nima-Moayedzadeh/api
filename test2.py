import cv2
import numpy as np
from sklearn import cluster
import time


params = cv2.SimpleBlobDetector_Params()

params.filterByInertia
params.minInertiaRatio = 0.6
diceEyes = [1]
detector = cv2.SimpleBlobDetector_create(params)


def get_blobs(frame):
    frame_blurred = cv2.medianBlur(frame, 7)
    frame_gray = cv2.cvtColor(frame_blurred, cv2.COLOR_BGR2GRAY)
    blobs = detector.detect(frame_gray)

    return blobs


def get_dice_from_blobs(blobs):
    # Get centroids of all blobs
    X = []
    for b in blobs:
        pos = b.pt

        if pos != None:
            X.append(pos)

    X = np.asarray(X)

    if len(X) > 0:
        # Important to set min_sample to 0, as a dice may only have one dot
        clustering = cluster.DBSCAN(eps=40, min_samples=0).fit(X)

        # Find the largest label assigned + 1, that's the number of dice found
        num_dice = max(clustering.labels_) + 1

        dice = []

        # Calculate centroid of each dice, the average between all a dice's dots
        for i in range(num_dice):
            X_dice = X[clustering.labels_ == i]

            centroid_dice = np.mean(X_dice, axis=0)

            dice.append([len(X_dice), *centroid_dice])
            diceEyes.append(len(X_dice))

        return (dice)

    else:
        return []


def overlay_info(frame, dice, blobs):
    # Overlay blobs
    for b in blobs:
        pos = b.pt
        r = b.size / 2

        cv2.circle(frame, (int(pos[0]), int(pos[1])),
                   int(r), (255, 0, 0), 2)

    # Overlay dice number
    for d in dice:
        # Get textsize for text centering
        textsize = cv2.getTextSize(
            str(d[0]), cv2.FONT_HERSHEY_PLAIN, 3, 2)[0]

        cv2.putText(frame, str(d[0]),
                    (int(d[1] - textsize[0] / 2),
                     int(d[2] + textsize[1] / 2)),
                    cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)


# Initialize a video feed
cap = cv2.VideoCapture(0)

#cap.set(3, 640)
#cap.set(4, 480)


def cameraStart():
    # Grab the latest image from the video feed
        start_time = time.time()
        img_counter = 0
        diceEyes.clear()
        while(True):
            ret, frame = cap.read()
        
        # We'll define these later
            blobs = get_blobs(frame)
            dice = get_dice_from_blobs(blobs)
            out_frame = overlay_info(frame, dice, blobs)

            cv2.imshow("frame", frame)
            if np.shape(frame) == (): # latest numpy / py3
                print ('no Image LOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOL')
        # fail !!
            res = cv2.waitKey(1)

            #Stop if the user presses "q"
            #if res & 0xFF == ord('q'):
                #return
            if time.time() - start_time >= 1.2: #<---- Check if 3 sec passed
                img_name = "opencv_frame_{}.png".format(img_counter)
                cv2.imwrite(img_name, frame)
                print("{} written!".format(img_counter))
                #start_time = time.time()
                img_counter += 1
            #if time.time() - start_time >= 5: #<---- Check if 3 sec passed    
        # When everything is done, release the capture
                #cap.release()
                cv2.destroyAllWindows()
                return
            