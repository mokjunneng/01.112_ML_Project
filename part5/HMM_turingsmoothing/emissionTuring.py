import sys


# Global variables
tags = []
obs = []
# unknown_word_count = 0
emission_count_dict = {}

def train(file=None, k=1):
    y_count_dict = {} 
    

    with open(file, "r",  encoding="utf-8") as f:
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
    
    return calc_e(y_count_dict, emission_count_dict, count_unknown(file))

# open test file and count unknown words
def count_unknown(filename):
    unknown_word_count = 0
    with open(filename, "r", encoding="utf-8") as f:
        for l in f.readlines():
            word = l.split()
            if (len(word) != 0):
                word_copy = word[0]
                if word not in obs:
                    word_copy = "#UNK#"
                    unknown_word_count += 1
    return unknown_word_count

#emission parameters    
def calc_e(y_count_dict, emission_count_dict, unknown_word_count):
    e = {}
    k = 1
    for (word, tag) in emission_count_dict.keys():
        e[(word, tag)] = emission_count_dict[(word, tag)] / (y_count_dict[tag] + k)
    
    # Handlde Unknown Words #
    for tag in y_count_dict.keys():
        n1 = 0 # number of words which appeared once with tag t
        N =  0 # total number of words which appeared with tag t
        for (key, value) in emission_count_dict.items():
            if (value == 1 and tag == key[1]):
                n1 +=1
            if (tag == key[1]): #key[1] is the tag
                N += 1
        e[("#UNK#", tag)] = n1/(unknown_word_count * N)
    return e


def get_words(emission_count_dict):
    words = []
    for (word, tag) in emission_count_dict.keys():
        words.append(word)
    return words
