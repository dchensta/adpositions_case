#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 22:10:08 2020

@author: Daniel Chen
"""

import csv

print("Queen Arya")

#READER
file = open("regulus_word_embdgs_list.tsv")
lines = csv.reader(file, delimiter='\t')

#WRITER
with open('r_converted_embdgs.csv', 'w', newline='') as csvfile :
    writer = csv.writer(csvfile)

    for line in lines :
        #positive tensor: 14 characters (including parentheses)
        #negative tensor: 15 characters (including parentheses)

        newRow = []
    
        for tensor in line :
            if len(tensor) == 14 : #positive tensor
                newTensor = tensor[7:13]
            else : #negative tensor: length = 15 to include negative sign -
                newTensor = tensor[7:14]
            newRow.append(newTensor)
        writer.writerow(newRow)