import sys


# Global variables
tags = []
obs = []

def train(file=None, k=1):
    y_count_dict = {} 
    emission_count_dict = {}

    with open(file, "r") as f:
        for line in f.readlines():
            if line == "\n":
                continue
            line = line.split()
            word = line[0]
            tag = line[len(line) - 1]
            if word not in obs:
                obs.append(word)
            if tag not in tags:
                tags.append(tag)
            # Count(y->x)
            if emission_count_dict.get((word, tag)):
                emission_count_dict[(word, tag)] += 1
            else:
                emission_count_dict[(word, tag)] = 1
            # Count(y)
            if y_count_dict.get(tag):
                y_count_dict[tag] += 1
            else:
                y_count_dict[tag] = 1
            
    
    return calc_e(y_count_dict, emission_count_dict, k)

def calc_e(y_count_dict, emission_count_dict, k):
    e = {}
    for (word, tag) in emission_count_dict.keys():
        e[(word, tag)] = emission_count_dict[(word, tag)] / (y_count_dict[tag] + k)
    # Handle UNK case
    for tag in y_count_dict.keys():
        e[("#UNK#", tag)] = k / (y_count_dict[tag] + k)  
    return e

# filename = "/Users/ganr/Desktop/ML/Project/SG/train" #change the file name here to run
# e_dict = train(filename)

def get_words(emission_count_dict):
    words = []
    for (word, tag) in emission_count_dict.keys():
        words.append(word)
    return words
