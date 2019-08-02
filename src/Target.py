from __future__ import division, absolute_import

import numpy as np
import pandas as pd
import random
from keras.preprocessing import sequence, text
from keras.utils import np_utils

np.random.seed(8)
random.seed(8)
    
class Target(object):
    """ Object class for targets - sequences of amino acids"""    
    def __init__(self, seq):
        self.seq = seq
        
# el diccionario deberia estar hecho de la concatenacion de todos los aminoacidos
    def creating_dict(self):
        """ Creates a dictionary from amino acids (string characters) to integer numbers
        from the sequence of a target. It returns the dictionary and the length of the dictionary """
        #Don't know why, but sequence is numpy.bytes_ type
        #sequence = self.seq.decode('UTF-8')
        #aminoacids = list(sequence)
        aminoacids = list(self.seq)
        unique_aas = sorted(list(set(aminoacids)))
        len_aas = len(unique_aas)
        #The "+1" is added in order not to have 0 in the dictionary - to mask padding zeros 
        aa_to_int = dict((c,i+1) for i,c in enumerate(unique_aas))
        return aa_to_int, len_aas
    
    def predefining_dict(self):
        list_aas = ['0', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
        nums = list(range(0, len(list_aas)))
        aa_to_int = dict(zip(list_aas, nums))
        return aa_to_int
    
    def stretching_seq(self, x,max_len):
        "Defining a function for the stretching padding, because its complexity"
        x_len = len(x)
        dif_len = max_len - x_len #Number of padding zeros 
        seq = "" #New encoding sequence to build
        if x_len > dif_len:
            xIndex = 0
            for i in range(dif_len):
                xChunk = round(x_len / (dif_len + 1))
                seq = "".join(string for string in [seq, x[xIndex:(xIndex+xChunk)], '0']) 
                xIndex = xIndex + xChunk
                x_len = x_len - xChunk
                dif_len = dif_len - 1
    
            seq = seq + x[xIndex]

        if  x_len <= dif_len:
            lastPad = round((dif_len) / (x_len))
            for c in x[:-1]:
                yPad = round(dif_len / (x_len-1))
                seq = "".join(string for string in [seq, c, "0"*yPad]) 
                x_len = x_len - 1
                dif_len = dif_len - yPad
            seq = seq + x[-1]
        return seq
    
    def stretching_seq_nonzeros(self, x,max_len):
        """This function is like stretch_padding but instead of padding with zeros, it pads with the closest amino acid"""
        x_len = len(x)
        dif_len = max_len - x_len #Number of padding zeros 
        seq = "" #New encoding sequence to build
        if x_len > dif_len:
            #x_len > dif_len
            xIndex = 0
            for i in range(dif_len):
                xChunk = round(x_len / (dif_len + 1))
                if xChunk==1:
                    seq = "".join(string for string in [seq, x[xIndex:(xIndex+xChunk)], x[(xIndex)]]) 
                else:
                    seq = "".join(string for string in [seq, x[xIndex:(xIndex+xChunk)], x[(xIndex+xChunk-1)]])
                xIndex = xIndex + xChunk
                x_len = x_len - xChunk
                dif_len = dif_len - 1
    
            seq = seq + x[xIndex]
        if  x_len <= dif_len:
            lastPad= round((dif_len)/(x_len))
            for idx, c in enumerate(x[:-1]):
                yPad = round(dif_len/(x_len-1))
                miniPad = int(yPad/2)
                if yPad%2 ==0:
                    seq = "".join(string for string in [seq, c, c*miniPad])
                    seq = "".join(string for string in [seq, x[idx+1]*miniPad])
                else:
                    if miniPad == 0:
                        seq = "".join(string for string in [seq, c, c])
                    else:
                        seq = "".join(string for string in [seq, c, c*miniPad])
                        seq = "".join(string for string in [seq, x[idx+1]*(miniPad+1)])
                x_len = x_len - 1
                dif_len = dif_len - yPad
            seq = seq + x[-1]
        return seq
    
    def padding_seq_position(self, max_len, pad_type="post"):
        """ Pads the sequence pre/post/mid/stretch/ext/rdm """
        prot_string = self.seq
        prot_len = len(prot_string)
        diflen = max_len - prot_len
        if pad_type=="pre":
            padded_seq = prot_string.zfill(max_len)
            return padded_seq
        elif pad_type=="mid":
            zeross = '0' * diflen
            firstpart, secondpart = prot_string[:int(prot_len/2)], prot_string[int(prot_len/2):]
            padded_seq = firstpart + zeross + ''.join(secondpart)
            return padded_seq
        elif pad_type=="post":
            padded_seq = prot_string.ljust(max_len, '0')
            return padded_seq
        elif pad_type =="strf":
            padded_seq = self.stretching_seq(prot_string, max_len)
            return padded_seq
        elif pad_type == "zoom":
            padded_seq = self.stretching_seq_nonzeros(prot_string, max_len)
            return padded_seq
        elif pad_type == "ext":
            zeross = '0' * int(diflen/2)
            if diflen%2 == 0:
                padded_seq = zeross + prot_string + zeross
            else:
                padded_seq = zeross + prot_string + zeross + "0"
            return padded_seq
        elif pad_type == "rnd":
            padded_seq = prot_string
            rdm_positions = np.random.randint(0,max_len+1,diflen)
            for i in rdm_positions:
                padded_seq = padded_seq[:i] + '0' + padded_seq[i:]
            return padded_seq
        else:
            print("Wrong padding value")
        
    def target_to_int(self, dictionary):
        """Translate a sequence of amino acids to a list of integers, given a dictionary"""
        target_int = []
        aminoacids = list(self.seq)
        for i in aminoacids:
            target_int.append(dictionary[i])
        return target_int
    
    def string_to_int(self, seq, dictionary):
        """Translate a sequence of amino acids to a list of integers, given a dictionary"""
        target_int = []
        aminoacids = list(seq)
        for i in aminoacids:
            target_int.append(dictionary[i])
        return target_int
    
    
    def int_to_onehot(self, x_int, num_classes):
        """Translate a list of integers to one hot encoding"""
        onehot = np_utils.to_categorical(x_int, num_classes=num_classes)
        return onehot
    
    def onehot_to_aa(self, onehot, diccionario):
        """Translate a list of one hot encoding to a sequence of amino acids given a dictionary"""
        result = []
        for i in onehot:
            num = np.argmax(i)
            if num == 0:
                continue
            result.append(list(diccionario.keys())[list(diccionario.values()).index(num)])
        return result
    
    def onehot_to_aa_dot(self, onehot, diccionario):
        """Translate a list of one hot encoding to a sequence of amino acids given a dictionary.
        When there is a 0 (padding), it writes a '.'"""
        result = []
        for i in onehot:
            num = np.argmax(i)
            if num == 0:
                result.append('0')
                continue
            result.append(list(diccionario.keys())[list(diccionario.values()).index(num)])
        return result
    
    def int_to_aa(self, x_int, diccionario):
        """Translate a list of integer numbers to a sequence of amino acids, given a dictionary"""
        result = []   
        for j in x_int:
            if j == 0:
                continue
            result.append(list(diccionario.keys())[list(diccionario.values()).index(j)])
        return result
    
    
    def max_target_length(self, targets_list):
        """Given a list of targets, finds the longest and returns its length"""
        #input must be an array of aminoacids
        max_len = 0
        for x in targets_list:
            if len(x.seq)>max_len:
                max_len = len(x.seq)     
        print ('The longest target consists of ', max_len, 'amino acids')
        return max_len
    
    def min_target_length(self, targets_list, thresh=10000):
        """Given a list of targets, finds the shortest and returns its length"""
        #input must be an array of aminoacids
        min_len = thresh
        for x in targets_list:
            if len(x.seq)<min_len:
                min_len = len(x.seq)     
        print ('The shortest target consists of ', min_len, 'amino acids')
        return min_len
    
    def concatenate_aas(self, target_list):
        """Given a target list, returns a sequence of concatenated amino acids. Useful for
        creating_dict"""
        target_seqs = [tg.seq for tg in target_list]
        sentence = ''.join(target_seqs)
        return sentence
    
# esta funcion tengo que extenderla para ver dónde pongo el padding
#    def padding_aas(self,list_seqs, max_len, padding='post'):
#        """Given a list of sequences, pad them with zeros"""
#        #max_len could be None if there is no need to specify
#        padded_x = sequence.pad_sequences(sequences=list_seqs, maxlen=max_len, padding=padding)
#        return padded_x
    
    def encode_as_blosum(self):
        """Given a sequence of amino acids, encode with BLOSUM matrix"""
        dict_blosum = {
        'A':[4, -1, -2, -2, 0, -1, -1, 0, -2, -1, -1, -1, -1, -2, -1, 1, 0, -3, -2, 0],
        'R':[-1, 5, 0, -2, -3, 1, 0, -2, 0, -3, -2, 2, -1, -3, -2, -1, -1, -3, -2, -3],
        'N':[-2, 0, 6, 1, -3, 0, 0, 0, 1, -3, -3, 0, -2, -3, -2, 1, 0, -4, -2, -3],
        'D':[-2, -2, 1, 6, -3, 0, 2, -1, -1, -3, -4, -1, -3, -3, -1, 0, -1, -4, -3, -3],
        'C':[0, -3, -3, -3, 9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1],
        'Q':[-1, 1, 0, 0, -3, 5, 2, -2, 0, -3, -2, 1, 0, -3, -1, 0, -1, -2, -1, -2],
        'E':[-1, 0, 0, 2, -4, 2, 5, -2, 0, -3, -3, 1, -2, -3, -1, 0, -1, -3, -2, -2],
        'G':[0, -2, 0, -1, -3, -2, -2, 6, -2, -4, -4, -2, -3, -3, -2, 0, -2, -2, -3, -3],
        'H':[-2, 0, 1, -1, -3, 0, 0, -2, 8, -3, -3, -1, -2, -1, -2, -1, -2, -2, 2, -3],
        'I':[-1, -3, -3, -3, -1, -3, -3, -4, -3, 4, 2, -3, 1, 0, -3, -2, -1, -3, -1, 3],
        'L':[-1, -2, -3, -4, -1, -2, -3, -4, -3, 2, 4, -2, 2, 0, -3, -2, -1, -2, -1, 1],
        'K':[-1, 2, 0, -1, -3, 1, 1, -2, -1, -3, -2, 5, -1, -3, -1, 0, -1, -3, -2, -2],
        'M':[-1, -1, -2, -3, -1, 0, -2, -3, -2, 1, 2, -1, 5, 0, -2, -1, -1, -1, -1, 1],
        'F':[-2, -3, -3, -3, -2, -3, -3, -3, -1, 0, 0, -3, 0, 6, -4, -2, -2, 1, 3, -1],
        'P':[-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7, -1, -1, -4, -3, -2],
        'S':[1, -1, 1, 0, -1, 0, 0, 0, -1, -2, -2, 0, -1, -2, -1, 4, 1, -3, -2, -2],
        'T':[0, -1, 0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1, 5, -2, -2, 0],
        'W':[-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1, -4, -3, -2, 11, 2, -3],
        'Y':[-2, -2, -2, -3, -2, -1, -2, -3, 2, -1, -1, -2, -1, 3, -3, -2, -2, 2, 7, -1],
        'V':[0, -3, -3, -3, -1, -2, -2, -3, -3, 3, 1, -2, 1, -1, -2, -2, 0, -3, -1, 4]}   
        result = []
        for i in self.seq:
            result.append(dict_blosum[i]) 
        return result