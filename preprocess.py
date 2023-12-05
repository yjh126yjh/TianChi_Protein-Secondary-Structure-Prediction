import numpy as np

seqdict={
    "G":[[0,22,30,34],-0.08,-0.88],
    "P":[[1,26,30,34],-0.32,-0.31],
    "T":[[2,27,31,34],-0.14,-0.25],
    "E":[[3,21,29,33],-0.7,0.15],
    "S":[[4,27,31,34],-0.16,-0.45],
    "K":[[5,25,32,35],-0.78,0.13],
    "C":[[6,28,30,34],0.5,-0.22],
    "L":[[7,22,30,34],0.76,-0.08],
    "M":[[8,28,30,34],0.38,0.18],
    "V":[[9,22,30,34],0.84,-0.28],
    "D":[[10,21,29,33],-0.7,-0.05],
    "A":[[11,22,30,34],0.36,-0.68],
    "R":[[12,25,32,35],-0.9,0.53],
    "I":[[13,22,30,34],0.9,-0.08],
    "N":[[14,23,31,34],-0.7,-0.07],
    "H":[[15,24,25,32,33],-0.64,0.26],
    "F":[[16,24,30,34],0.56,0.40],
    "W":[[17,24,30,34],-0.18,0.96],
    "Y":[[18,24,31,34],-0.26,0.63],
    "Q":[[19,23,31,34],-0.7,0.13],
    "X":[[20],-0.04,0]
}

secdict={
    " ":0,
    "E":1,
    "T":2,
    "S":3,
    "H":4,
    "G":5,
    "B":6,
    "I":7,
}

newdict = {'A': [1, 3, 5, 6],
           'R': [1, 3, 5, 7, 20, 27, 28, 30, 32, 34, 35],
           'N': [1, 3, 5, 7, 16, 18],
           'D': [1, 3, 5, 7, 17, 18],
           'C': [1, 3, 5, 9],
           'Q': [1, 3, 5, 7, 20, 25],
           'E': [1, 3, 5, 7, 20, 26],
           'G': [0, 3, 5],
           'H': [1, 3, 5, 7, 38],
           'I': [1, 3, 5, 10, 12, 14, 23],
           'L': [1, 3, 5, 7, 19, 22, 23],
           'K': [1, 3, 5, 7, 20, 31, 33],
           'M': [1, 3, 5, 7, 21, 24, 29],
           'F': [1, 3, 5, 7, 39],
           'P': [1, 3, 5, 7, 37],
           'S': [1, 3, 5, 8],
           'T': [1, 3, 5, 11, 13, 15],
           'W': [1, 3, 5, 7, 40],
           'Y': [1, 3, 5, 7, 36, 41],
           'V': [1, 3, 5, 10, 12, 13],
           'X': []}

res=np.zeros((20000,700, 88))
i_num=0
with open(r"data_seq_train.txt") as f:
    for lines in f:
        i_char = 0
        for char in lines[:-1]:
            indexlist = seqdict.get(char)
            if indexlist == None:
                indexlist = seqdict.get("X")
            for index in indexlist[0]:
                res[i_num][i_char][index] = 1
            res[i_num][i_char][36] = indexlist[1]
            res[i_num][i_char][37] = indexlist[2]
            indexlist2=newdict.get(char)
            if indexlist2==None:
                indexlist2=newdict.get("X")
            for index in indexlist2:
                res[i_num][i_char][index+38]=1
            i_char+=1
        i_num+=1

i_num=0
with open(r"data_sec_train.txt") as f:
    for lines in f:
        i_char = 0
        for char in lines[:-1]:
            index=secdict.get(char)+80
            res[i_num][i_char][index]=1
            i_char+=1
        i_num+=1

with open("res.npy","wb") as f:
    np.save(f,res)


print("end")