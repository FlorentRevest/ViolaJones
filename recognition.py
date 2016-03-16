#!/usr/bin/python3

from lxml import etree
import os,sys
from PIL import Image

def integrate(im) :
    pix = im.load()
    width , height = im.size
    for x in range(0,width-1) :
        for y in range(0,height-1) :
            if x == 0 and y != 0 :
                pix[x,y] += pix[x,y-1] 
            elif y == 0  and x != 0:
                pix[x,y] += pix[x-1,y]
            elif x != 0 and y != 0 :
                pix[x,y] += pix[x-1,y] + pix[x,y-1] - pix[x-1,y-1]
                
im = Image.open(sys.argv[1]).convert("I")
integrate(im)

tree = etree.parse("haarcascade_frontalface_default.xml").getroot()
cascade = tree.find("cascade")
height = int(cascade.find("height").text)
width = int(cascade.find("width").text)
stages = cascade.find("stages")

for stage in stages :
    weakClassifiers = stage.find("weakClassifiers")
    for node in weakClassifiers :
        print (node.tag)



    
