#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 10:58:57 2020

@author: Daniel Chen
"""

print("Queen Daenerys")
import csv

with open("pp_annotations_master.csv") as file:
    reader = csv.reader(file)

    caseCount = dict(); annCount = dict()
    for row in reader:
        case = row[4]; ann = row[5]

        if case not in caseCount :
            caseCount[case] = 1
        else :
            caseCount[case] += 1

        if ann not in annCount :
            annCount[ann] = 1
        else :
            annCount[ann] += 1

print("caseCount: \n", caseCount.items())
print("Number of unique cases: ", len(caseCount))
print("annCount: \n", annCount.items())
print("Number of unique annotations: ", len(annCount))