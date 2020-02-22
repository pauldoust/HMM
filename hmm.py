import numpy as np
from hmmlearn import hmm
def Question1():
    states = ["Rainy", "Cloudy", "Sunny"]
    n_states = len(states)

    observations = ["Museum", "Beach"]
    n_observations = len(observations)
    model = hmm.MultinomialHMM(n_components=n_states, init_params="",
    n_iter=10,algorithm='map',tol=0.00001)
    model.startprob_ = np.array([1,0,0])
    model.transmat_ = np.array([
    [0.2, 0.6, 0.2],
    [0.3, 0.2, 0.5],
    [0.1, 0.1, 0.8]
    ])
    model.emissionprob_ = np.array([
    [0.7, 0.3],
    [1, 0],
    [0.1, 0.9]])
    gen1 = model.sample(3)
    print(gen1)
    seqgen2, stat2= model.sample(5)
    print(seqgen2)
    print(stat2)
    gen3=model.sample(2)
    print(gen3)

    sequence1 = np.array([[0,1]]).T
    logproba=model.score(sequence1)
    print(logproba)
    print(np.exp(logproba))
    logproba_extend=model.score_samples(sequence1)
    print(logproba_extend)
    p = model.predict(sequence1)
    print(p)
    p_extend = model.score_samples(sequence1)


    sequence3 = np.array([[1, 1, 1, 1]]).T
    sequence4 = np.array([[0, 1, 0, 1, 1]]).T
    sample = np.concatenate([sequence3, sequence4])
    lengths = [len(sequence3), len(sequence4)]
    model.fit(sample,lengths)

    sequence = np.array([[0, 1, 0, 1]]).T
    logprob, state_seq = model.decode(sequence, algorithm="viterbi")

    print("Observations", ", ".join(map(lambda x: observations[int(x)], sequence)))
    print("Associated states:", ", ".join(map(lambda x: states[x], state_seq)))



def Question2():
    #1. Compute the probability of the string abbaa
    states = ["State_1", "State_2", "State_3"]
    n_states = len(states)

    observations = ["O1", "O2", "O3"]
    n_observations = len(observations)
    model = hmm.MultinomialHMM(n_components=n_states,
    n_iter=10,algorithm='map',tol=0.00001)
    model.startprob_ = np.array([0.5, 0.3, 0.2])
    model.transmat_ = np.array([
    [0.45, 0.35, 0.20],
    [0.10, 0.50, 0.40],
    [0.15, 0.25, 0.60]
    ])
    model.emissionprob_ = np.array([
    [1, 0],
    [0.5, 0.5],
    [0, 1]])

    sequence1 = np.array([[0,1,1,0,0]]).T
    logproba=model.score(sequence1)
    print("log probability of the string abbaa: ", logproba)
    print("probability of the string abbaa: ", np.exp(logproba))
    
    #############################################################################
    
    #2. Apply BaumWelch with only one iteration and check the probability of the string
    model = hmm.MultinomialHMM(n_components=n_states, init_params="",
    n_iter=1,algorithm='map',tol=0.00001)
    model.startprob_ = np.array([0.5, 0.3, 0.2])
    model.transmat_ = np.array([
    [0.45, 0.35, 0.20],
    [0.10, 0.50, 0.40],
    [0.15, 0.25, 0.60]
    ])
    model.emissionprob_ = np.array([
    [1, 0],
    [0.5, 0.5],
    [0, 1]])

    model.fit(sequence1)
    p_extend = model.score(sequence1)
    print("log One Iteration BaumWelch: ", p_extend)
    print("One Iteration BaumWelch: ", np.exp(p_extend))
    ###############################################################################
    #3. Do the same thing after 15 iterations
    ###############################################################################
    model = hmm.MultinomialHMM(n_components=n_states, init_params="",
    n_iter=15,algorithm='map',tol=0.00001)
    model.startprob_ = np.array([0.5, 0.3, 0.2])
    model.transmat_ = np.array([
    [0.45, 0.35, 0.20],
    [0.10, 0.50, 0.40],
    [0.15, 0.25, 0.60]
    ])
    model.emissionprob_ = np.array([
    [1, 0],
    [0.5, 0.5],
    [0, 1]])

    model.fit(sequence1)
    p_extend = model.score(sequence1)
    print("log 15 Iterations BaumWelch: ", (p_extend))
    print("15 Iterations BaumWelch: ", np.exp(p_extend))
    ###############################################################################
    #4. Try to obtain the result at convergence
    ###############################################################################
    model4 = hmm.MultinomialHMM(n_components=n_states, init_params="",
    n_iter=150,algorithm='map',tol=0.00000001)
    model4.startprob_ = np.array([0.5, 0.3, 0.2])
    model4.transmat_ = np.array([
    [0.45, 0.35, 0.20],
    [0.10, 0.50, 0.40],
    [0.15, 0.25, 0.60]
    ])
    model4.emissionprob_ = np.array([
    [1, 0],
    [0.5, 0.5],
    [0, 1]])

    model4.fit(sequence1)
    p_extend = model4.score(sequence1)
    print("log At convergence BaumWelch: ", (p_extend))
    print("At convergence BaumWelch: ", np.exp(p_extend))
    ###############################################################################
    #5. Now create an HMM with 5 states with parameters initialized at any non zero correct values.
    ###############################################################################
    model = hmm.MultinomialHMM(n_components=5, init_params="",
    n_iter=120,algorithm='map',tol=0.00000001)
    model.startprob_ = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
    model.transmat_ = np.array([
    [0.40, 0.30, 0.10, 0.10, 0.10],
    [0.5, 0.10, 0.30, 0.05, 0.05],
    [0.10, 0.20, 0.60, 0.05 , 0.05],
    [0.30, 0.40, 0.10, 0.10, 0.10],
    [0.25, 0.25, 0.25, 0.10 , 0.15]
    ])
    model.emissionprob_ = np.array([
    [1, 0],
    [0.5, 0.5],
    [0.5, 0.5],
    [0.5, 0.5],
    [0, 1]])

    model.fit(sequence1)
    p_extend = model.score(sequence1)
    print("log At 5 states BaumWelch: ", (p_extend))
    print("At 5 states BaumWelch: ", np.exp(p_extend))


def Question3():
    ########################################################
    #1. Create HMM with the library
    ########################################################
    states = ["State_1", "State_2"]
    n_states = len(states)

    observations = ["O1", "O2"]
    n_observations = len(observations)
    model = hmm.MultinomialHMM(n_components=n_states, init_params="",
    n_iter=50,algorithm='map',tol=0.00001)
    model.startprob_ = np.array([0.31, 0.69])
    model.transmat_ = np.array([
    [ 0.40, 0.60],
    [ 0.52, 0.48]
    ])
    model.emissionprob_ = np.array([
    [0.49, 0.51],
    [0.40, 0.60]])

    ########################################################
    #2.Learn the HMM with the following sample L1 = faaabb; abaabbb; aaababb; aabab; abg}.
    ########################################################
    sequence1 = np.array([[0,0,0,1,1]]).T
    sequence2 = np.array([[0,1,0,0,1,1,1]]).T
    sequence3 = np.array([[0,0,0,1,0,1,1]]).T
    sequence4 = np.array([[0,0,1,0,1]]).T
    sequence5 = np.array([[0,1]]).T
    sample = np.concatenate([sequence1, sequence2, sequence3, sequence4, sequence5])
    print("sample: ", sample)
    lengths = [len(sequence1), len(sequence2), len(sequence3), len(sequence4), len(sequence5)]
    model.fit(sample,lengths)
    #Moel obtained after training:
    print(model.transmat_)
    print(model.startprob_)
    print(model.emissionprob_)
    #######################################################
    #3
    #######################################################
    states = ["State_1", "State_2"]
    n_states = len(states)

    observations = ["O1", "O2"]
    n_observations = len(observations)
    model2 = hmm.MultinomialHMM(n_components=n_states, init_params="",
    n_iter=50,algorithm='map',tol=0.00001)
    model2.startprob_ = np.array([0.31, 0.69])
    model2.transmat_ = np.array([
    [ 0.40, 0.60],
    [ 0.52, 0.48]
    ])
    model2.emissionprob_ = np.array([
    [0.49, 0.51],
    [0.40, 0.60]])

    sequence1 = np.array([[1,1,1,0,0]]).T
    sequence2 = np.array([[1,0,1,1,0,0]]).T
    sequence3 = np.array([[1,1,1,0,1,0,0]]).T
    sequence4 = np.array([[1,1,0,1,1,0]]).T
    sequence5 = np.array([[1,1,0,0]]).T
    sample = np.concatenate([sequence1, sequence2, sequence3, sequence4, sequence5])
    lengths = [len(sequence1), len(sequence2), len(sequence3), len(sequence4), len(sequence5)]
    model2.fit(sample,lengths)
    print(model2.transmat_)
    print(model2.startprob_)
    print(model2.emissionprob_)

    #######################################################
    #5 Compute the probabilities of the strings aababbb and bbabaaa. Are the results intuitive?
    #######################################################
    sequence_1 = np.array([[0, 0, 1, 0, 1, 1, 1]]).T
    sequence_2 = np.array([[1, 1, 0, 1, 0, 0, 0]]).T
    p_extend = model.score(sequence1)
    # print("log prob for first model first sequence: ", (p_extend))
    print("prob for first model first sequence: ", np.exp(p_extend))

    p_extend = model.score(sequence2)
    # print("log prob for first model second sequence: ", (p_extend))
    print("prob for first model second sequence: ", np.exp(p_extend))


    p_extend = model2.score(sequence1)
    # print("log prob for second model first sequence: ", (p_extend))
    print("prob for second model fist sequence: ", np.exp(p_extend))



    p_extend = model2.score(sequence2)
    # print("log prob for second model second sequence: ", (p_extend))
    print("prob for second model second sequence: ", np.exp(p_extend))

Question1()
Question2()
Question3()