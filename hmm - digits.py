import numpy as np
from hmmlearn import hmm
def Number0():
    states = [ "extra_1", "extra_2", "extra_3", "extra_4","extra_1", "extra_2", "extra_3", "extra_4","extra_1", "extra_2", "extra_3", "extra_4"]
    n_states = len(states)
    observations = ["North", "North_East", "East", "South_East","South", "South_West","West","North_West"]
    n_observations = len(observations)
    sample,lengths,tst_sample,tst_lengths = process_data(r"F:\MLDM\2nd Semester\Done_ML\Practical 2\digit_strings\0-sample.txt")
    model = hmm.MultinomialHMM(n_components=18,n_iter=200,algorithm='map' ,tol=0.001, init_params='ste')
    model.fit(sample, lengths)
    zero = np.array([[2,2,2,3,3,3,4,3,4,3,4,4,4,5,4,6,5,6,6,5,6,6,6,6,6,6,6,7,6,7,0,0,0,0,0,1,0,0,1,1,1,2,2,2,1]]).T
    one = np.array([[2,3,2,3,3,4,6,6,6,7,6,5,6,5,5,3,3,3,4,4,4,4,4,6,5,6,6,6,6,6,7,0,2,2,2,2,1,0,7,7,7,7,6,7,0,0,0,1,2,3,3,2,1,1,1,2,1]]).T
    logproba = model.score(zero)
    print("log probability of the string abbaa: ", logproba)
    print("probability of the string abbaa: ", np.exp(logproba))
    
    logproba1 = model.score(one)
    print("1log probability of the string abbaa: ", logproba1)
    print("1probability of the string abbaa: ", np.exp(logproba1))



def process_data(fileName):
    f = open(fileName, "r")
    f1 = f.readlines()
    trn_dataset = None
    trn_lengths = []
    tst_dataset = None
    tst_length = []
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
        # counter = counter + 1
        if counter == 3:
            counter =0



    return trn_dataset,trn_lengths, tst_dataset,tst_length
Number0()