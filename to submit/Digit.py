from sklearn.externals import joblib
import numpy as np
from hmmlearn import hmm

class Digit:
    def __init__(self, name, n_states, n_iter = 100,  observations = None):
        self.name = name
        self.n_states = n_states
        self.n_iter = n_iter
        self.observations = observations
        self.model = hmm.MultinomialHMM(n_components=self.n_states, n_iter = self.n_iter, algorithm='map' ,tol=0.001, init_params='ste')
        self.load()

    def train(self, dataset, lengths):
        self.model.fit(dataset, lengths)
        return self.model

    def prob(self, sequence, lengths):
        return np.exp(self.model.score(sequence, lengths))

    def save(self):
        joblib.dump(self.model, self.name +".pkl")

    def load(self):
        self.model = joblib.load(self.name+ ".pkl") 
        print("Model: ", self.name, " was loaded successfully.")
