from os import path
import numpy
import glob

import matplotlib.pyplot as plt
from pylab import *

#number of classes(40 or 41)
NCLASS = 40

#IOU Threshold
THRESH = 0.55

itemList = [
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

conversionIDTable = [
0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]


#if active this list, ID order change to ARC Official List order.
"""
conversionIDTable = [
0, 5, 2, 7, 39, 8, 9, 14, 13, 19, 15,
32, 27, 34, 33, 29, 24, 18, 23, 11, 4,
20, 38, 28, 31, 1, 6, 12, 35, 10, 22,
21, 36, 25, 3, 30, 26, 40, 37, 16, 17, 41]
"""

def readTxt(filePath):
    coordinate = []
    f = open(filePath, 'r')

    if f != None:
        for row in f:
            data = row.split()
            coordinate.append(data)
        f.close()
        return coordinate
    else:
        print("[ERROR] Can't read:" + filePath)
        return 1


if __name__ == "__main__":
    #matching reesults path
    resultPath = "./results/matchingResults"

    #output(totalresult, confusionMatrix) path
    outPath = "./"

    fileList = glob.glob(resultPath + "/*.txt")
    fileList.sort()

    totaldataCnt = 0    # number of all(detected + not detected) boxes
    totalmatchCnt = 0   # number of true class boxes
    totalhitCnt = 0     # number of detected boxes
    totalmissCnt = 0    # number of false class boxes
    totalnohitCnt = 0   # number of not detected boxes
    totalIou = 0.0

    confusionMat = [[0 for i in range(NCLASS)] for j in range(NCLASS)]

    for filePath in fileList:
        dataCnt = 0
        matchCnt = 0
        hitCnt = 0
        missCnt = 0
        nohitCnt = 0

        resultData = readTxt(filePath)

        for i in range(0, len(resultData)):
            dataCnt += 1
            if(resultData[i][0]!='0' and resultData[i][1]!='0'):
                #detected
                hitCnt += 1
                totalIou += float(resultData[i][2])
                confusionMat[conversionIDTable[int(resultData[i][1])]-1][conversionIDTable[int(resultData[i][0])]-1] += 1
                if(float(resultData[i][2]) >= THRESH):
                    #IOU >= Threshold
                    if (resultData[i][0]==resultData[i][1]):
                        #true class
                        matchCnt += 1
                    else:
                        #false class
                        missCnt += 1
                else:
                    #low IOU
                    missCnt += 1
            else:
                #not detected
                nohitCnt += 1

        if dataCnt != (hitCnt+nohitCnt):
            print("Error1")
            print(dataCnt,hitCnt,nohitCnt)
            sys.exit()
        if hitCnt != (matchCnt+missCnt):
            print("Error2")
            print(hitCnt,matchCnt,missCnt)
            sys.exit()

        totaldataCnt += dataCnt
        totalmatchCnt += matchCnt
        totalhitCnt += hitCnt
        totalmissCnt += missCnt
        totalnohitCnt += nohitCnt

    print("Total Result")
    print("Matching Rate: "+str(float(totalmatchCnt)/float(totalhitCnt)))
    print("Miss(No detection box) Rate: "+str(float(totalnohitCnt)/float(totaldataCnt)))
    print("Mean IoU:"+ str(totalIou/totalhitCnt))


    #confusionMatrix normalization
    normMat = []
    for i in confusionMat:
        a = 0
        tmpMat = []
        a = sum(i,0)
        for j in i:
            if a == 0:
                tmpMat.append(0.0)
            else:
                tmpMat.append(float(j)/float(a))
        normMat.append(tmpMat)

    #draw confusionMatrix
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(array(normMat), cmap=cm.jet, interpolation='nearest')
    cb = fig.colorbar(res)
    cb.ax.set_yticklabels(["0%","10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"])
    confusionMat = numpy.array(confusionMat)
    width, height = confusionMat.shape
    plt.xticks(range(width), itemList[:width],rotation =90)
    plt.yticks(range(height), itemList[:height])
    plt.tick_params(labelsize=7)
    plt.subplots_adjust(left=0.05, bottom=0.10, right=0.95, top=0.95)
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")

    for i in range(0, NCLASS):
        print(str(i+1) + ": " + str(normMat[i][i]))

    #file output
    f = open(outPath + "/totalresult.txt", 'w')
    f.writelines("Total Result" + '\n')
    f.writelines("Matching Rate: "+str(float(totalmatchCnt)/float(totalhitCnt)) + '\n')
    f.writelines("Miss(No detection box) Rate: "+str(float(totalnohitCnt)/float(totaldataCnt)) + '\n')
    f.writelines("Mean IoU:"+ str(totalIou/totalhitCnt) + '\n')
    for i in range(0, NCLASS):
        f.writelines(str(i+1) + ": " + str(normMat[i][i]) + '\n')

    f.close()

    savefig(outPath + "/confusionMatrix.pdf", format="pdf")
    savefig(outPath + "/confusionMatrix.png", format="png")
