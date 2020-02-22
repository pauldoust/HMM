import numpy as np

def load_data(fileName):
    f = open(fileName, "r")
    f1 = f.readlines()
    trn_dataset = None
    trn_lengths = []
    tst_dataset = None
    tst_length = []
    tst_lines = []
    counter = 0
    for x in f1:
        s = ",".join( x[i] for i in range(0, len(x)-1))
        nparray = np.fromstring(s,dtype=int, sep=",")
        nparray = np.array([nparray]).T
        if counter < 2:        
            if trn_dataset is None:
                trn_dataset = nparray
            else:
                trn_dataset = np.concatenate([trn_dataset, nparray])
            trn_lengths.append(len(nparray))
        else:
            if tst_dataset is None:
                tst_dataset = nparray
            else:
                tst_dataset = np.concatenate([tst_dataset, nparray])
            tst_length.append(len(nparray))
        
            tst_lines.append(nparray)

        counter = counter + 1
        if counter == 3:
            counter = 0

    return trn_dataset,trn_lengths, tst_dataset,tst_length, tst_lines