# -*- coding:utf-8 -*-

import os

count = 0

com = 0

inCom = False

num_file = 0

for root,dirs,files in os.walk(os.path.join(os.getcwd(), "src")):
    for file in files:
        fn = os.path.join(root,file)
        if "ext\\" in fn:
            continue
        if "gui" in fn:
            continue
        if "tests" in fn:
            continue
        if "stats_line_num.py" in fn:
            continue
        if "jitify" in fn:
            continue
        if "sdk_pt" in fn:
            continue
        if ".natvis" in fn:
            continue
        try:
            # print(file)
            
            f = open(fn, "r")
            count += len(f.readlines())
        except :
            print(fn)

        
        
        num_file += 1


# print(count, num_file)
        

a = 0b1 << 3

def size(num):
    ret = 0
    for i in range(0, 4):

        if num % 2 != 0:
            ret += 1
        print(num, i, num % 2)
        num = num >> 1

        
    return ret

(size(0b0010))