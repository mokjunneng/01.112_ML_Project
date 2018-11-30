import numpy as np
from collections import defaultdict
import math

global_tags = ["START", "STOP"]
global_words = []
training_instances = []  # all the sentences in training set

def extract_feature_vector(file):
    transition_count_dict = defaultdict(float)
    emission_count_dict = defaultdict(float)
    y_count_dict = defaultdict(float)

    with open(file, "r", encoding="utf-8") as f:
        previous_tag = "START"
        current_words = []
        current_tags = []
        for line in f.readlines():
            tag = ""
            if line == "\n":
                tag = "STOP"
                training_instances.append((current_words, current_tags))
                current_words = []
                current_tags = []
            else:
                line = line.split()
                word = line[0]
                tag = line[-1]
                # Record down the list of words and tags that appear in the dataset
                if word not in global_words:
                    global_words.append(word)
                if tag not in global_tags:
                    global_tags.append(tag)
                # Record down all the counts of word given tag
                
                emission_count_dict[(word, tag)] += 1

                current_words.append(word)
                current_tags.append(tag)
    
            # Record down all the transitions between tag
            transition_count_dict[(previous_tag, tag)] += 1
            y_count_dict[tag] += 1

            if line == "\n":
                # start of a new training instance
                previous_tag = "START"
            else:
                previous_tag = tag
    return emission_count_dict, transition_count_dict                                                                                        

class StructuredPerceptron(object):
    def __init__(self, emission_counts, transition_counts):
        self.feature_weights = defaultdict(float)
        self.emission_counts = emission_counts
        self.transition_counts = transition_counts
                    
    def fit(self, train_data, no_of_epochs=5, learning_rate=0.2):
        for epoch in range(no_of_epochs):
            for i, (words, tags) in enumerate(train_data):
                predicted_tags = self.decode(words)
                # Update weights
                for j, tag in enumerate(predicted_tags):
                    if j == 0:
                        previous_tag = "START"
                        previous_tag_true = "START"
                    else:
                        previous_tag = predicted_tags[j-1]
                        previous_tag_true = tags[j-1]
                    if tag != tags[j]:
                        self.feature_weights[(previous_tag_true, tags[j])] += learning_rate * self.transition_counts[(previous_tag_true, tags[j])]
                        self.feature_weights[(previous_tag, tag)] -= learning_rate * self.transition_counts[(previous_tag, tag)]
                    
                        self.feature_weights[(words[j], tags[j])] += learning_rate * self.emission_counts[(words[j], tags[j])]
                        self.feature_weights[(words[j], tag)] -= learning_rate * self.emission_counts[(words[j], tag)]

    def decode(self, words):
        """
        Viterbi algorithm for decoding
        :param words:
        """
        best_score = {}
        best_edge = {}

        best_score[(0, "START")] = 0
        best_edge[(0, "START")] = None
        for i in range(len(words)):
            for current_tag in global_tags:
                for prev_tag in global_tags:
                    if best_score.get((i, prev_tag)) != None and self.transition_counts[(prev_tag, current_tag)]:
                        score = best_score[(i, prev_tag)] + \
                            self.feature_weights[(prev_tag, current_tag)] * self.transition_counts[(prev_tag, current_tag)] + \
                            self.feature_weights[(words[i], current_tag)] * self.emission_counts[(words[i], current_tag)]
                        if best_score.get((i+1, current_tag)) == None or best_score.get((i+1, current_tag)) < score:
                            best_score[(i+1, current_tag)] = score
                            best_edge[(i+1, current_tag)] = (i, prev_tag)
        # Handle the last layer "STOP"
        for tag in global_tags:
            if best_score.get((len(words), tag)) != None and self.transition_counts[(tag, "STOP")]:
                score = best_score[(len(words), tag)] + self.feature_weights[(tag, "STOP")]*self.transition_counts[(tag, "STOP")]
                if best_score.get((len(words) + 1, "STOP")) == None or best_score.get((len(words) + 1, "STOP")) < score:
                    best_score[(len(words) + 1, "STOP")] = score
                    best_edge[(len(words) + 1, "STOP")] = (len(words), tag)
                    
        predicted_tags = [""] * len(words)
        curr_tag = "STOP"
        for i in reversed(range(len(words) + 1)):
            if i == 0:
                continue
            _ , best_tag = best_edge[(i+1, curr_tag)]
            predicted_tags[i-1] = best_tag
            curr_tag = best_tag
        return predicted_tags

    def predict(self, file):
        with open(file, 'r', encoding="utf-8") as f:
            test_data = f.read().rstrip().split('\n\n')
        tag_sequences = []
        for sequence in test_data:
            word_array = sequence.splitlines()
            predicted_tags = self.decode(word_array)
            tag_sequences.append([(word_array[i], predicted_tags[i]) for i in range(len(word_array))])
        self.create_test_result_file(tag_sequences, "dev.p5.out")

    def create_test_result_file(self, test_result, filename):
        with open(filename, "w",  encoding="utf-8") as f:
            for sequence in test_result:
                for word, tag in sequence:
                    f.write(f"{word} {tag}\n")
                f.write("\n")

if __name__ == "__main__":
    emission_counts, transition_counts = extract_feature_vector("EN/train")
    sp = StructuredPerceptron(emission_counts, transition_counts)
    sp.fit(training_instances)
    sp.predict("EN/dev.in")

