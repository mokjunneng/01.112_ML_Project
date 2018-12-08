from collections import defaultdict


global_tags = ["START", "STOP"]
global_words = []

def get_training_data(file):
    training_instances = []  # all the sentences in training set
    tags = set(["START"])
    with open(file, "r", encoding="utf-8") as f:
        current_words = []
        current_tags = []
        for line in f.readlines():
            if line == "\n":
                training_instances.append((current_words, current_tags))
                current_words = []
                current_tags = []
            else:
                line = line.split()
                word = line[0]
                tag = line[-1]
                tags.add(tag)
                current_words.append(word)
                current_tags.append(tag)
    return training_instances, list(tags)


def get_emission_transition_tag_count(file):
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