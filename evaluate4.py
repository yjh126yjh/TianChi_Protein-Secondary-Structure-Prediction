import numpy as np
import model4

# onehot for 21 amino acid & features
seqdict = {
    "G": [[0, 22, 30, 34], -0.08, -0.88],
    "P": [[1, 26, 30, 34], -0.32, -0.31],
    "T": [[2, 27, 31, 34], -0.14, -0.25],
    "E": [[3, 21, 29, 33], -0.7, 0.15],
    "S": [[4, 27, 31, 34], -0.16, -0.45],
    "K": [[5, 25, 32, 35], -0.78, 0.13],
    "C": [[6, 28, 30, 34], 0.5, -0.22],
    "L": [[7, 22, 30, 34], 0.76, -0.08],
    "M": [[8, 28, 30, 34], 0.38, 0.18],
    "V": [[9, 22, 30, 34], 0.84, -0.28],
    "D": [[10, 21, 29, 33], -0.7, -0.05],
    "A": [[11, 22, 30, 34], 0.36, -0.68],
    "R": [[12, 25, 32, 35], -0.9, 0.53],
    "I": [[13, 22, 30, 34], 0.9, -0.08],
    "N": [[14, 23, 31, 34], -0.7, -0.07],
    "H": [[15, 24, 25, 32, 33], -0.64, 0.26],
    "F": [[16, 24, 30, 34], 0.56, 0.40],
    "W": [[17, 24, 30, 34], -0.18, 0.96],
    "Y": [[18, 24, 31, 34], -0.26, 0.63],
    "Q": [[19, 23, 31, 34], -0.7, 0.13],
    "X": [[20], -0.04, 0]
}

# onehot for 8 structure
secdict = {
    " ": 0,
    "E": 1,
    "T": 2,
    "S": 3,
    "H": 4,
    "G": 5,
    "B": 6,
    "I": 7,
}
table = [' ', 'E', 'T', 'S', 'H', 'G', 'B', 'I']

filepath = r"test.txt"  # <---- filepath of amino acid sequence
n = len(open(filepath, 'r').readlines())

X_test = np.zeros((n, 700, 38))
Y_test = np.zeros((n, 700, 1))
i_num = 0

# preprocess
with open(filepath) as f:
    for lines in f:
        i_char = 0
        for char in lines[:-1]:
            indexlist = seqdict.get(char)
            if indexlist == None:
                indexlist = seqdict.get("X")
            for index in indexlist[0]:
                X_test[i_num][i_char][index] = 1
            X_test[i_num][i_char][36] = indexlist[1]
            X_test[i_num][i_char][37] = indexlist[2]
            Y_test[i_num][i_char][0] = 1
            i_char += 1
        i_num += 1

net = model4.CNN_model()

# load Weights
net.load_weights(r"whole_sequence-best.hdf5")

predictions = net.predict(X_test)

# postprocess and output results
with open("result.txt", 'w') as f:
    for i in range(Y_test.shape[0]):  # process each sequence
        line = ""
        for j in range(Y_test.shape[1]):  # process each character of the sequence
            if np.sum(Y_test[i, j, :]) == 0:  # np.sum(Y_test[i, j, :]) == 0 if it is padding
                break
            index = np.argmax(predictions[i, j, :])
            line += table[index]
        line += "\n"
        f.write(line)
