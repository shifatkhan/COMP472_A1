# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 16:21:35 2020

@author: NgoWi
"""

import matplotlib
import math

data1 = (0,1)
data2 = (1,0)
data3 = (-1,0)
data4 = (0,-1)
data5 = (0.5, 0.5)
data6 = (-0.5, -0.5)
data7 = (-0.5, 0.5)
data8 = (0.5, -0.5)
data9 = (4,4)
data10 = (-4, -4)
data11 = (-4, 4)
data12 = (4, -4)
data13 = (4, 0)
data14 = (-4, 0)
data15 = (0, 4)
data16 = (0, -4)

alldata = (data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13, data14, data15, data16)

def euc(c1,c2, x, y):
    return math.sqrt( math.pow((c1 - x), 2) + math.pow((c2 - y), 2))

def kmeans(datalist, c1, c2):
    changed = True
    array_pre = []
    array_post = []
    c11 = c1
    c12 = c2
    while changed:
        array_post = []
        for tup in datalist:
            t1 = euc(tup[0], tup[1], c11[0], c11[1])
            t2 = euc(tup[0], tup[1], c12[0], c12[1])
            
            if t1 <= t2:
                array_post.append("C1")
            else:
                array_post.append("C2")
                
        print("PREE: ", array_pre)
        print("POST: ", array_post)
        print("")
        if len(array_pre) != 0:
            changed = array_pre != array_post
        
        if changed:
            a1 = []
            a2 = []
            for x in range(16):
                if array_post[x] == "C1":
                    a1.append(alldata[x])
                else:
                    a2.append(alldata[x])
            
            newc11 = 0
            newc12 = 0
            
            newc21 = 0
            newc22 = 0
            for t in a1:
                newc11 += t[0]
                newc12 += t[1]
            newc11 = newc11/len(a1)
            newc12 = newc12/len(a1)
            
            for t2 in a2:
                newc21 += t2[0]
                newc22 += t2[1]
            newc21 = newc21/len(a2)
            newc22 = newc22/len(a2)
            
            c11 = (newc11, newc12)
            c12 = (newc21, newc22)
            array_pre = array_post.copy()
            
        else:
            print("FINAL ", array_post)

c11 = (0,0)
c12 = (4,4)
c21 = (-5,0)
c22 = (2,0)
kmeans(alldata, c21, c22)
