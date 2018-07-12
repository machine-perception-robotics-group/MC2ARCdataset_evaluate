import os
from os import path
import numpy
import cv2
import glob

#image extension(png, jpg, bmp ...etc)
IMAGE_EXT = "png"

#wait time for cv.wait (if 0 then no wait)
WAITTIME = 0

#if your detection results have classlabel(ex: DVD, avery_binder...etc), set 1
LABEL_FLAG = 0

#if your detection results has normalized, set 1
NORMALIZED = 1

#if teach label have category and color classification, set 1
# ! need not change
CAT_PASS_FLAG = 0

COLOR_TABLE = [
[   0,    0,    0],
[  85,    0,    0],
[ 170,    0,    0],
[ 255,    0,    0],
[   0,   85,    0],
[  85,   85,    0],
[ 170,   85,    0],
[ 255,   85,    0],
[   0,  170,    0],
[  85,  170,    0],
[ 170,  170,    0],
[ 255,  170,    0],
[   0,  255,    0],
[  85,  255,    0],
[ 170,  255,    0],
[ 255,  255,    0],
[   0,    0,   85],
[  85,    0,   85],
[ 170,    0,   85],
[ 255,    0,   85],
[   0,   85,   85],
[  85,   85,   85],
[ 170,   85,   85],
[ 255,   85,   85],
[   0,  170,   85],
[  85,  170,   85],
[ 170,  170,   85],
[ 255,  170,   85],
[   0,  255,   85],
[  85,  255,   85],
[ 170,  225,   85],
[ 255,  255,   85],
[   0,    0,  170],
[  85,    0,  170],
[ 170,    0,  170],
[ 255,    0,  170],
[   0,   85,  170],
[  85,   85,  170],
[ 170,   85,  170],
[ 255,   85,  170],
[   0,  170,  170],
[ 255,  255,  255]]

itemIDList = [
"0 BG",
"1",
"2",
"3",
"4",
"5",
"6",
"7",
"8",
"9",
"10",
"11",
"12",
"13",
"14",
"15",
"16",
"17",
"18",
"19",
"20",
"21",
"22",
"23",
"24",
"25",
"26",
"27",
"28",
"29",
"30",
"31",
"32",
"33",
"34",
"35",
"36",
"37",
"38",
"39",
"40",
"41"]

def convNormalizedCord(data, HEIGHT, WIDTH):
    x = float(data[1]) * WIDTH
    y = float(data[2]) * HEIGHT
    w = float(data[3]) * WIDTH
    h = float(data[4]) * HEIGHT
    x1 = x - (w/2.)
    y1 = y - (h/2.)
    x2 = x + (w/2.)
    y2 = y + (h/2.)
    return(int(data[0]), int(x1), int(y1), int(x2), int(y2))

def convResCord(data, HEIGHT, WIDTH):
    if NORMALIZED == 1:
        data = convNormalizedCord(data, HEIGHT, WIDTH)
    if int(data[0]) == 0:
        classID = 41
    else:
        classID = int(data[0])
    return(classID, int(data[1]), int(data[2]), int(data[3]), int(data[4]) )

def getIOU(boxA, boxB):
    #if length between boxAcenter and boxBcenter is too far, return 0
    center_boxA = numpy.array([(boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0])
    center_boxB = numpy.array([(boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0])
    if numpy.linalg.norm(center_boxA - center_boxB) >= 500:
        return 0

    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = (xB - xA + 1) * (yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou

def readTxt(filePath, dType):
    coordinate = []
    f = open(filePath, 'r')

    if f != None:
        for row in f:
            data = row.split()
            if(LABEL_FLAG==1 and dType=="result"):
                data = [data[0], data[2], data[3], data[4], data[5]]
            elif( dType == "result"):
                data = [data[0], data[1], data[2], data[3], data[4]]
            elif(CAT_PASS_FLAG==1):
                data = [data[0], data[3], data[4], data[5], data[6]]
            coordinate.append(data)
        f.close()
        return coordinate
    else:
        print("[ERROR] Can't read:" + filePath)
        return 1

def drawBB(img, data):
    color = [ COLOR_TABLE[int(data[0])][2], COLOR_TABLE[int(data[0])][1], COLOR_TABLE[int(data[0])][0] ]
    height, width = img.shape[:2]
    x1 = data[1]
    y1 = data[2]
    x2 = data[3]
    y2 = data[4]
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    cv2.putText(img, itemIDList[int(data[0])], (x1, y1-2), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
    cv2.putText(img, itemIDList[int(data[0])], (x1, y1), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,0), 1)
    return img

def drawAllBB(img, coordinate):
    for data in coordinate:
        img = drawBB(img, data)
    return img

if __name__ == "__main__":
    imagePath = "./image/"
    teachPath = "./teach/"
    resultPath = "./results/"

    #mkdir for Matching Result
    os.mkdir(resultPath+"/matchingResults")

    fileList = glob.glob(resultPath + "*.txt")
    fileList.sort()

    for filePath in fileList:
        print filePath

        resultData = readTxt(filePath, "result")
        teachData = readTxt(teachPath + filePath[filePath.rfind('/'):-4] + ".txt", "teach")
        img = cv2.imread(imagePath + filePath[filePath.rfind('/'):-4] + "." + IMAGE_EXT)
        HEIGHT, WIDTH = img.shape[:2]
        imgBackup = img.copy()

        #Convert to Full(teach) coordinates
        for i in range(0, len(teachData)):
            teachData[i] = convNormalizedCord(teachData[i], HEIGHT, WIDTH)

        #Convert for result coordinates
        for i in range(0, len(resultData)):
            resultData[i] = convResCord(resultData[i], HEIGHT, WIDTH)

        #init result lists
        hit = [False] * len(teachData)
        success = [False] * len(resultData)
        maxIouList = [0] * len(resultData)
        trueCategory = [0] * len(resultData)

        #search for box which have highest value of IoU
        for j in range(0, len(resultData)):
            maxIou = 0.0
            maxIndex = 0
            for i in range(0, len(teachData)):
                iou = getIOU(teachData[i][1:], resultData[j][1:])
                if(maxIou<iou)and(iou<=1.0):
                    maxIou = iou
                    maxIndex = i
                img = imgBackup.copy()
                img = drawBB(img, teachData[i])
                img = drawBB(img, resultData[j])
                if WAITTIME != 0:
                    cv2.imshow("", img)
                    cv2.waitKey(WAITTIME)

            #found
            maxIouList[j] = maxIou
            img = imgBackup.copy()
            if( maxIou>0.35 ):
                hit[maxIndex] = True
                success[j] = teachData[maxIndex][0]==resultData[j][0]
                trueCategory[j] = teachData[maxIndex][0]
                img = drawBB(img, teachData[maxIndex])
                img = drawBB(img, resultData[j])
                cv2.putText(img, "Max IoU:" + str(maxIou), (0, 25) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
                cv2.putText(img, "Class Match:" + str(success[j]), (0, 50) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
            else:
                img = drawBB(img, resultData[j])
                cv2.putText(img, "No Matching", (0, 25) , cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 2)
            if WAITTIME != 0:
                cv2.imshow("", img)
                cv2.waitKey(WAITTIME*5)

        print("Result")
        print("Matching Success")
        for i in range(0, len(resultData)):
            if(success[i]==True):
                print(str(resultData[i]))

        print("Matching Failed")
        for i in range(0, len(resultData)):
            if(success[i]==False):
                print(str(resultData[i]) + str(maxIouList[i]))

        print("No Match")
        for i in range(0, len(teachData)):
            if(hit[i]==False):
                print(str(teachData[i]))

        print("File output")
        f = open(resultPath + "matchingResults/" + filePath[filePath.rfind('/'):], 'w')
        for i in range(0, len(resultData)):
            writeData = str(resultData[i][0]) + " " + str(trueCategory[i]) + " " + str(maxIouList[i]) + '\n'
            f.writelines(writeData)
        #miss boxes
        for i in range(0, len(teachData)):
            if(hit[i]==False):
                writeData = "0" + " " + str(teachData[i][0]) + " 0.0" + '\n'
                f.writelines(writeData)
        f.close()

    print("Done")

    cv2.destroyAllWindows()
