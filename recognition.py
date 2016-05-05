#!/usr/bin/python3

from lxml import etree
import os,sys
from PIL import Image

def integrate(im) :
    pix = im.load()
    width , height = im.size
    for x in range(width) :
        for y in range(height) :
            if x == 0 and y != 0 :
                pix[x,y] += pix[x,y-1] 
            elif y == 0  and x != 0:
                pix[x,y] += pix[x-1,y]
            elif x != 0 and y != 0 :
                pix[x,y] += pix[x-1,y] + pix[x,y-1] - pix[x-1,y-1]
                
def sumPixel(x,y,width,height,im) :
    pix=im.load()
    return pix[x+width,y+height] + pix[x,y] - pix[x+width,y] - pix[x,y+height]

im = Image.open(sys.argv[1]).convert("I")
integrate(im)
imageWidth, imageHeight = im.size

tree = etree.parse("haarcascade_frontalface_default.xml").getroot()
cascade = tree.find("cascade")
windowHeight = int(cascade.find("height").text)
windowWidth = int(cascade.find("width").text)
stages = cascade.find("stages")
features = cascade.find("features")

windowY = 0
windowX = 0
while windowY + windowHeight < imageHeight :
    stagePassed = True
    i = 0
    for stage in stages:
        i = i+1
        print(i)
        stageValue = 0
        stageThreshold = float(stage.find("stageThreshold").text)
        weakClassifiers = stage.find("weakClassifiers")        
        for node in weakClassifiers :
            internalNodes = node.find("internalNodes").text
            leftChildIndex, rightChildIndex, featureIndex, nodeThreshold = (n for n in internalNodes.split())
            featureIndex = int(featureIndex)
            nodeThreshold = float(nodeThreshold)
            leafValues = node.find("leafValues").text        
            leftLeafValue, rightLeafValue = (float(n) for n in leafValues.split())
            nodeValue = 0
            rects = features[featureIndex].find("rects")
            for rectNode in rects :
                rect = rectNode.text
                x,y,width,height,weight = (n for n in rect.split())
                nodeValue += float(weight)*sumPixel(int(x),int(y),int(width),int(height),im) 
            if nodeValue > nodeThreshold : # right
                stageValue += rightLeafValue
            else :                         # left
                stageValue += leftLeafValue

        if stageValue < stageThreshold:
            stagePassed = False
            break;

    if stagePassed:
        print("passed")

    windowX += 10
    if windowX + windowWidth > imageWidth :
        windowY += 10
        windowX = 0
