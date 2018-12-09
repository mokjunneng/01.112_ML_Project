from utils import get_training_data
import sys
import yaml
from collections import defaultdict, Counter
import numpy as np
# from numpy import empty, zeros, ones, log, exp, add, int32
from math import exp, log
import time

class CRF(object):
    def __init__(self, tags, train_data):
        self.feature_weights = defaultdict(float)
        self.tags = tags
        self.train_data = train_data

    def train(self, iterations=5, learning_rate=0.2):
        train_data = self.train_data
        w = self.feature_weights
        for i in range(iterations):
            start = time.clock()
            rate = 10 / (i+1)**0.501
            for words, tags in train_data:
                for feature, val in self.get_expectation(words).items():
                    w[feature] -= rate * val + 0.1 * w[feature]
                for feature in self.get_all_features(words, tags).keys():
                    w[feature] += rate - 0.1 * w[feature]
            end = time.clock()
            print(f"Trained {i+1}-th iteration for {end - start} seconds")
     
    def predict(self, test_file, outfile):
        test_data = self.read_test_file(test_file)
        test_words_tags_sequence = []
        for sequence in test_data:
            word_array = sequence.splitlines()
            predicted_tags = self.viterbi(word_array)
            test_words_tags_sequence.append([(word_array[i], predicted_tags[i]) for i in range(len(word_array))])
        self.create_test_result_file(test_words_tags_sequence, outfile)
    
    def viterbi(self, words):
        """
        Viterbi algorithm for decoding
        """
        N = len(words)
        M = len(self.tags)
        tags = list(self.tags)

        best_score = {}
        best_edge = {}

        # First layer
        for tag in tags:
            features = self.get_features(words[0], tag, "START")
            feature_weights = sum((self.feature_weights[x] * count for x, count in features.items()))
            # Best scores for first layer is simply weight(START -> <Tag>)
            best_score[(1, tag)] = feature_weights

        for i in range(1, N):

            for current_tag in tags:

                curr_best_score = float('-Inf')

                for prev_tag in tags:
                        
                        features = self.get_features(words[i], current_tag, prev_tag)
                        feature_weights = sum((self.feature_weights[x] * count for x, count in features.items()))

                        score = best_score[(i, prev_tag)] + feature_weights

                        if score > curr_best_score:
                            best_score[(i+1, current_tag)] = score
                            best_edge[(i+1, current_tag)] = (i, prev_tag)
                            curr_best_score = score

        # Backtrack   
        predicted_tags = [""] * len(words)
        curr_tag = ""
        last_layer_best_score = float('-Inf')
        for tag in tags:
            score = best_score[(N, tag)]
            if score > last_layer_best_score:
                last_layer_best_score = score
                curr_tag = tag
        predicted_tags[N-1] = curr_tag
        
        for i in reversed(range(1, N)):
            _ , best_tag = best_edge[(i+1, curr_tag)]
            predicted_tags[i-1] = best_tag
            curr_tag = best_tag
    
        return predicted_tags
        
    def get_expectation(self, words):
        """Expectation term"""
        N = len(words)
        M = len(self.tags)

        global_features = self.get_features_for_all_possible_sequences(words)

        g0, g = self.get_all_g(words)
        forwardprobs = self.get_forward(g0, g, N)
        backwardprobs = self.get_backward(g, N)

        logZ = self.logsumexp(forwardprobs[N-1, :])
        expectation = defaultdict(float)

        # Expectation at the first position of sequence
        e = np.exp(g0 + backwardprobs[0, :] - logZ)
        for m in range(M):
            posterior_prob = e[m]
            for feature in global_features[0][0][m]:
                expectation[feature] += posterior_prob

        for n in range(1, N):
            e = np.exp((np.add.outer(forwardprobs[n-1, :], backwardprobs[n, :]) + g[n-1, :, :] - logZ))
            for mprev in range(M):
                for m in range(M):
                    posterior_prob = e[mprev, m]
                    for feature in global_features[n][mprev][m]:
                        expectation[feature] += posterior_prob
        return expectation

    def get_features_for_all_possible_sequences(self, words):
        tags = self.tags
        M = len(tags)
        N = len(words)
        feature_counts = [[[0 for k in range(M)] for j in range(M)] for i in range(N)]
        for x, word in enumerate(words):
            if x == 0:
                for y, tag in enumerate(tags):
                    feature_count_local = Counter()
                    feature_count_local.update([feature for feature in self.feature_functions(word, tag, "START").values()])
                    feature_counts[x][0][y] = feature_count_local
            for y, tag in enumerate(tags):
                for yprev, prev_tag in enumerate(tags):
                    feature_count_local = Counter()
                    feature_count_local.update([feature for feature in self.feature_functions(word, tag, prev_tag).values()])
                    feature_counts[x][yprev][y] = feature_count_local
        return feature_counts

    def get_features(self, word, curr_tag, prev_tag):
        feature_counts = Counter()
        feature_counts.update([feature for feature in self.feature_functions(word, curr_tag, prev_tag).values()])
        return feature_counts
    
    def get_all_features(self, words, tags):
        feature_counts = Counter()
        for x in words:
            for i, tag in enumerate(tags):
                if i == 0:
                    prev_tag = "START"
                else:
                    prev_tag = tags[i-1]
                feature_counts.update([feature for feature in self.feature_functions(x, tag, prev_tag).values()])
        return feature_counts

    def feature_functions(self, word, tag, prev_tag):
        word_lower = word.lower()
        features = {
            "word_lower": f"WORD_LOWER_{word_lower}",
            "word_suffix3": f"WORD_SUFFIX3_{word_lower[-3:]}",
            "word_suffix2": f"WORD_SUFFIX2_{word_lower[-2:]}",
            "word_prefix3": f"WORD_PREFIX3_{word_lower[:3]}",
            "word_prefix2": f"WORD_PREFIX2_{word_lower[:2]}",
            "word_is_title": f"WORD_IS_TITLE_{word.istitle()}",
            "word_is_lower": f"WORD_IS_LOWER_{word.islower()}",
            "word_tag": f"WORD_{word}_TAG_{tag}",
            "prevtag_tag": f"TAG_{tag}_PREVTAG_{prev_tag}"
        }
        return features

    def get_all_g(self, words):
        """
        g(prev_y, y) = sum(weights of feature func)
        This function calculates the log of it
        """

        N = len(words)
        tags = list(self.tags)
        M = len(tags)

        g0 = np.empty(M)
        g = np.empty((N-1, M, M))
        # Initialize first layer
        for i, tag in enumerate(tags):
            g0[i] = sum([self.feature_weights[f] * count for f, count in self.get_features(words[0], tag, "START").items()]) 

        for i in range(1, N):
            for j, curr_tag in enumerate(tags):
                for k, prev_tag in enumerate(tags):
                    g[i-1, k, j] = sum([self.feature_weights[f] * count for f, count in self.get_features(words[i], curr_tag, prev_tag).items()]) 
        return (g0, g)
    
    def get_forward(self, g0, g, N):
        """
        Calculate matrix of forward log probabilities

        forward[i, y] = log of the sum of scores of all sequences for 0 to position i
                      = forward[i-1, y_prev] * g[n, y_prev, y]
        """
        M = len(self.tags)
        forward = np.zeros((N, M))
        forward[0, :] = g0

        for n in range(1, N):
            prev_nodes = forward[n-1, :]
            for m in range(M):
                forward[n, m] = self.logsumexp(prev_nodes + g[n-1, :, m])
        return forward

    def get_backward(self, g, N):
        """
        Calculate matrix of backward log probabilities

        forward[i, y] = log of the sum of scores of all sequences for 0 to position i
                      = forward[i-1, y_prev] * g[n, y_prev, y]
        """
        M = len(self.tags)
        backward = np.zeros((N, M))
        for n in reversed(range(N-1)):
            next_nodes = backward[n+1, :]
            for m in range(M):
                backward[n, m] = self.logsumexp(next_nodes + g[n, m, :])
        
        return backward
    
    def logsumexp(self, a):
        """Log sum exp trick to solve underflow issue"""
        b = a.max()
        return b + np.log((np.exp(a - b)).sum())

    def read_test_file(self, file):
        with open(file, 'r', encoding="utf-8") as f:
            test_data = f.read().rstrip().split('\n\n')
        return test_data
    
    def create_test_result_file(self, test_result, filename):
        with open(filename, "w",  encoding="utf-8") as f:
            for sequence in test_result:
                for word, tag in sequence:
                    f.write(f"{word} {tag}\n")
                f.write("\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("Usage on Windows:  python emission.py [train file] [dev.in file]")
        print ("Usage on Linux/Mac:  python3 emission.py [train file] [dev.in file]")
        sys.exit()
    train_data, tags = get_training_data(sys.argv[1])
    crf = CRF(tags, train_data)
    crf.train(iterations=10)
    crf.predict(sys.argv[2], sys.argv[3])
