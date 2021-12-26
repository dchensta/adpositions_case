#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 01:27:03 2020

@author: Daniel Chen
"""
print("Steve Rogers <3")

import csv

txtRef = open("regulus_labels.tsv")
reader1 = csv.reader(txtRef)

with open("pp_annotations_dos.csv", 'w', newline ='') as csvfile:
    writer = csv.writer(csvfile)

    with open("pp_annotations_5.csv") as ch5 :
        reader = csv.reader(ch5)

        count = 0
        for row in reader :
            if row[5] != "" :
                print(row[1],": ",row[5])
                writer.writerow(row)
                count += 1
        print("Count for chapter 5: ", count, " annotations")


txtRef.close()