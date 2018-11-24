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
            word, tag = line.split()
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
        e[("UNK", tag)] = k / (y_count_dict[tag] + k)  
    return e

def test(params, file=None):
    tag_sequences = []
    with open(file, "r") as f:
        tag_sequence = []
        for l in f.readlines():
            if l == "\n":
                tag_sequences.append(tag_sequence)
                tag_sequence = []
                continue
            word = l.split()[0]
            word_copy = word
            if word not in obs:
                word_copy = "UNK"
            y_arg_max = 0
            tag_arg_max = ""
            for tag in tags:
                y_arg = params[(word_copy, tag)] if params.get((word_copy, tag)) else 0
                if y_arg > y_arg_max:
                    tag_arg_max = tag
            tag_sequence.append((word, tag_arg_max))
    return tag_sequences

def create_test_result_file(test_result, filename):
    with open(filename, "w") as f:
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

    e_dict = train(sys.argv[1])
    tag_sequences = test(e_dict, sys.argv[2])
    create_test_result_file(tag_sequences, "dev.p2.out")
