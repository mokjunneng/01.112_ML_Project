import numpy as np
from collections import defaultdict, Counter
import math
import random
import sys       
from utils import get_training_data                                                                              

class StructuredPerceptron(object):
    def __init__(self):
        self.feature_weights = defaultdict(float)
        self.feature_weights_average = defaultdict(float)
        # self.emission_counts = emission_counts
        # self.transition_counts = transition_counts
        self.tags = set()
                    
    def fit(self, train_data, no_of_epochs=5, learning_rate=0.2):
        for epoch in range(no_of_epochs):

            correct = 0
            total = 0
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
                    for feature, count in prediction_features.items():
                        self.feature_weights[feature] = self.feature_weights[feature] - learning_rate * count
                    
                correct += sum([1 for (predicted, gold) in zip(predicted_tags, tags) if predicted == gold])
                total += len(tags)
                
            print(f"Training accuracy : {correct/total}")
            random.shuffle(train_data)

    def get_global_features(self, words, tags):
        feature_counts = Counter()
        for i, (word, tag) in enumerate(zip(words, tags)):
            previous_tag = "START" if i == 0 else tags[i-1]
            feature_counts.update(self.get_features(word, tag, previous_tag))
        return feature_counts

    def get_features(self, word, tag, previous_tag):
        features = [
            (previous_tag, tag),
            (word, tag)
        ]
        return features
    
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
        :param words:
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
        print ("Usage on Windows:  python emission.py [train file] [dev.in file]")
        print ("Usage on Linux/Mac:  python3 emission.py [train file] [dev.in file]")
        sys.exit()

    train_data = get_training_data(sys.argv[1])

    sp = StructuredPerceptron()
    sp.fit(train_data, no_of_epochs=5, learning_rate=0.2)
    sp.predict(sys.argv[2], sys.argv[3])
    # Train multiple models of different hyperparameter values - no. of epochs, learning rate
    # for i in range(1, 10, 2):
    #     for j in range(1, 6, 2):
    #         sp = StructuredPerceptron()
    #         sp.fit(train_data, no_of_epochs=i, learning_rate=j*0.1)
    #         sp.predict(sys.argv[2], f"EN-Result/dev.p5.perceptron.{i}.{j*0.1}.out")

