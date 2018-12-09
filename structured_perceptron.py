import numpy as np
from collections import defaultdict, Counter
import math
import random
import sys       
from utils import get_training_data   
import time    
import re                                                                       

class StructuredPerceptron(object):
    def __init__(self):
        self.feature_weights = defaultdict(float)
        self.feature_weights_average = defaultdict(float)
        self.total_feature_weights = defaultdict(tuple)
        # self.emission_counts = emission_counts
        # self.transition_counts = transition_counts
        self.tags = set()
        self.i = 0
                    
    def fit(self, train_data, no_of_epochs=5, learning_rate=0.2):
        for epoch in range(no_of_epochs):

            correct = 0
            total = 0
            start = time.clock()
            print(f"Training epoch {epoch+1} with learning rate {learning_rate} ....")

            for i, (words, tags) in enumerate(train_data):

                for tag in tags:
                    self.tags.add(tag)

                # get prediction
                predicted_tags = self.decode(words)

                gold_features = self.get_global_features(words, tags)
                prediction_features = self.get_global_features(words, predicted_tags)
                
                # Update weights
                if predicted_tags != tags:
                    for feature, count in gold_features.items():
                        self.feature_weights[feature] = self.feature_weights[feature] + learning_rate * count
                    # self.update(gold_features, i, learning_rate)
                    for feature, count in prediction_features.items():
                        self.feature_weights[feature] = self.feature_weights[feature] - learning_rate * count
                    
                correct += sum([1 for (predicted, gold) in zip(predicted_tags, tags) if predicted == gold])
                total += len(tags)
                
            print(f"Training accuracy : {correct/total}")
            # random.shuffle(train_data)
            end = time.clock()
            print(f"Time taken for {epoch + 1}th iteration: {end - start} seconds")
        # self.average()
    
    def update(self, features, iteration, learning_rate):
        self.i += 1
        for f, count in features.items():
            w = self.feature_weights[f]
            if not self.total_feature_weights[f]:
                w_iteration, total_weight = (0, 0)
            else:
                w_iteration, total_weight = self.total_feature_weights[f]
            # Update weight sum with last registered weight since it was updated
            total_weight += (self.i - w_iteration) * w
            w_iteration = self.i
            total_weight += learning_rate * count

            # Update weight and total
            self.feature_weights[f] += learning_rate * count
            self.total_feature_weights[f] = (w_iteration, total_weight)

    def average(self):
        for f, w in self.feature_weights.items():
            if not self.total_feature_weights[f]:
                w_iteration, total_weight = (0, 0)
            else:
                w_iteration, total_weight = self.total_feature_weights[f]
            # Update weight sum with last registered weight since it was updated
            total_weight += (self.i - w_iteration) * w
            averaged_w = total_weight / self.i if self.i else 0
            self.feature_weights[f] = averaged_w

            w_iteration = 0
            total_weight = averaged_w
            self.total_feature_weights[f] = (w_iteration, total_weight)

    def get_global_features(self, words, tags):
        feature_counts = Counter()
        for i, (word, tag) in enumerate(zip(words, tags)):
            previous_tag = "START" if i == 0 else tags[i-1]
            feature_counts.update(self.get_features(word, tag, previous_tag))
        return feature_counts

    def get_features(self, word, tag, previous_tag):
        word_lower = word.lower()
        prefix3 = word_lower[:3]
        # prefix2 = word_lower[:2]
        suffix3 = word_lower[-3:]
        # suffix2 = word_lower[-2:]

        features = [
                    'TAG_%s' % (tag),                       # current tag
                    'TAG_BIGRAM_%s_%s' % (previous_tag, tag),  # tag bigrams
                    'WORD+TAG_%s_%s' % (word, tag),            # word-tag combination
                    'WORD_LOWER+TAG_%s_%s' % (word_lower, tag),# word-tag combination (lowercase)
                    'UPPER_%s_%s' % (word[0].isupper(), tag),  # word starts with uppercase letter
                    'DASH_%s_%s' % ('-' in word, tag),         # word contains a dash
                    # 'PREFIX3_%s' % (prefix3),
                    # 'PREFIX2_%s' % (prefix2),
                    # 'SUFFIX3_%s' % (suffix3),
                    # 'SUFFIX2_%s' % (suffix2),                    
                    'PREFIX3+TAG_%s_%s' % (prefix3, tag),        # prefix3 and tag
                    'SUFFIX3+TAG_%s_%s' % (suffix3, tag),        # suffix3 and tag
                    # 'PREFIX2+TAG_%s_%s' % (prefix2, tag),        # prefix2 and tag
                    # 'SUFFIX2+TAG_%s_%s' % (suffix2, tag),        # suffix2 and tag
                    'WORD+TAG_BIGRAM_%s_%s_%s' % (word, tag, previous_tag),
                    'SUFFIX3+2TAGS_%s_%s_%s' % (suffix3, previous_tag, tag),
                    'PREFIX3+2TAGS_%s_%s_%s' % (prefix3, previous_tag, tag),
                    'WORDSHAPE_%s_TAG_%s' % (self.shape(word), tag)
                    # 'SUFFIX2+2TAGS_%s_%s_%s' % (suffix2, previous_tag, tag),
                    # 'PREFIX2+2TAGS_%s_%s_%s' % (prefix2, previous_tag, tag)
                ]
        return features

    def shape(self, word):
        result = []
        for char in word:
            if char.isupper():
                result.append('X')
            elif char.islower():
                result.append('x')
            elif char in '0123456789':
                result.append('d')
            else:
                result.append(char)
        return re.sub(r"x+", "x*", ''.join(result))
    
    # def fit_average(self, train_data, no_of_epochs=5, learning_rate=0.2):
    #     for epoch in range(no_of_epochs):
    #         for i, (words, tags) in enumerate(train_data):
    #             predicted_tags = self.decode(words)
    #             # Update weights
    #             for j, tag in enumerate(predicted_tags):
    #                 if j == 0:
    #                     previous_tag = "START"
    #                     previous_tag_true = "START"
    #                 else:
    #                     previous_tag = predicted_tags[j-1]
    #                     previous_tag_true = tags[j-1]
    #                 if tag != tags[j]:
    #                     self.feature_weights[(previous_tag_true, tags[j])] += learning_rate * self.transition_counts[(previous_tag_true, tags[j])]
    #                     self.feature_weights[(previous_tag, tag)] -= learning_rate * self.transition_counts[(previous_tag, tag)]
                    
    #                     self.feature_weights[(words[j], tags[j])] += learning_rate * self.emission_counts[(words[j], tags[j])]
    #                     self.feature_weights[(words[j], tag)] -= learning_rate * self.emission_counts[(words[j], tag)]

    #                     self.feature_weights_average[(previous_tag_true, tags[j])] += self.feature_weights[(previous_tag_true, tags[j])]
    
    def decode(self, words):
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
            feature_weights = sum((self.feature_weights[x] for x in features))
            # Best scores for first layer is simply weight(START -> <Tag>)
            best_score[(1, tag)] = feature_weights

        for i in range(1, N):

            for current_tag in tags:

                curr_best_score = float('-Inf')

                for prev_tag in tags:
                        
                        features = self.get_features(words[i], current_tag, prev_tag)
                        feature_weights = sum((self.feature_weights[x] for x in features))

                        score = best_score[(i, prev_tag)] + feature_weights

                        if score > curr_best_score:
                            best_score[(i+1, current_tag)] = score
                            best_edge[(i+1, current_tag)] = (i, prev_tag)
                            curr_best_score = score

        # Last layer
        curr_best_score = float('-Inf')
        for tag in tags: 
            feature_weights = self.feature_weights[(tag, "STOP")]
            score = best_score[(N, tag)] + feature_weights
            if score > curr_best_score:
                best_score[(N+1, "STOP")] = score
                best_edge[(N+1, "STOP")] = (N, tag)
                curr_best_score = score

        # Backtrack   
        predicted_tags = [""] * len(words)
        curr_tag = "STOP"
        for i in reversed(range(1, len(words) + 1)):
            _ , best_tag = best_edge[(i+1, curr_tag)]
            predicted_tags[i-1] = best_tag
            curr_tag = best_tag
    
        return predicted_tags

    def predict(self, file, outfile):
        with open(file, 'r', encoding="utf-8") as f:
            test_data = f.read().rstrip().split('\n\n')
        tag_sequences = []
        for sequence in test_data:
            word_array = sequence.splitlines()
            predicted_tags = self.decode(word_array)
            tag_sequences.append([(word_array[i], predicted_tags[i]) for i in range(len(word_array))])
        
        self.create_test_result_file(tag_sequences, outfile)

    def create_test_result_file(self, test_result, filename):
        with open(filename, "w",  encoding="utf-8") as f:
            for sequence in test_result:
                for word, tag in sequence:
                    f.write(f"{word} {tag}\n")
                f.write("\n")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("Usage on Windows:  python emission.py [train file] [dev.in file] [result filepath]")
        print ("Usage on Linux/Mac:  python3 emission.py [train file] [dev.in file] [result filepath]")
        sys.exit()

    train_data = get_training_data(sys.argv[1])

    sp = StructuredPerceptron()
    sp.fit(train_data, no_of_epochs=50, learning_rate=0.2)
    sp.predict(sys.argv[2], sys.argv[3])
