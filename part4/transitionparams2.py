#part 4a
import sys, os

def estimateTransition(train_file):
    """ Read a train file and returns a q parameters dictionary, tags"""
    counts_tuv_dict = {} #count transition from t,u --> v
    counts_tu_dict = {} #number of occurence for each tag pair (t,u)
    
    # read train file and fill in tag_full_list
    tag_full_list = read_train_file(train_file)

    # count transition of tags
    counts_tu_dict, counts_tuv_dict = count(tag_full_list)

    # get transition params
    q = get_transition_params(counts_tu_dict, counts_tuv_dict)

    tags = get_tags(counts_tu_dict)

    return q, tags

def read_train_file(train_file):
    """ Read train file & return a 2d tag_full_list """
    change_of_sentence_flag = 0 #a marker for the end of sentence
    tag_full_list = [[] for i in range(1000000)]
    tag_full_list[0].append('START')
    sentence_counter = 0
    
    for line in train_file:
        if line.strip(): # if not empty
            # print(line)
            if change_of_sentence_flag == 1: # first tag in a sentence
                tag_full_list[sentence_counter].append('START')
                change_of_sentence_flag = 0

            tag = line.strip().split(" ")[-1]
            # print(tag)
            tag_full_list[sentence_counter].append(tag)

        else: # empty line
            if change_of_sentence_flag == 0: # stop of the sentence
                tag_full_list[sentence_counter].append('STOP')
            change_of_sentence_flag = 1
            sentence_counter += 1
    tag_full_list = [x for x in tag_full_list if x != []] # strip away empty list
    # print(tag_full_list)
    return tag_full_list

def count(tag_full_list):
    """ Count tag transition & return (counts_tu_dict, counts_tuv_dict) """
    counts_tuv_dict = {} #count transition from t,u --> v
    counts_tu_dict = {} #number of occurence for each tag pair (t,u)

    for i in range(len(tag_full_list)): # sentence i
        if len(tag_full_list[i]) >= 3: # ignore sentence less than 3 words
            # start counting from index of 2
            for j in range(2,len(tag_full_list[i])): # tag j
                tag_t = tag_full_list[i][j-2]
                tag_u = tag_full_list[i][j-1]
                tag_v = tag_full_list[i][j]

                # count number of occurence for each tag pair (t,u)
                if ((tag_t,tag_u) not in counts_tu_dict):
                    counts_tu_dict[(tag_t,tag_u)] = 1
                else:
                    counts_tu_dict[(tag_t,tag_u)] += 1

                # count number of occurence for each tag sequence (t,u,v)
                if ((tag_t,tag_u,tag_v) not in counts_tuv_dict):
                    counts_tuv_dict[(tag_t,tag_u,tag_v)] = 1
                else:
                    counts_tuv_dict[(tag_t,tag_u,tag_v)] += 1
                
    return counts_tu_dict, counts_tuv_dict

def get_transition_params(counts_tu_dict, counts_tuv_dict):
    """ Take in counts_tu_dict, counts_tuv_dict & returns a q transition parameters dictionary"""
    q = {}
    for (t,u,v) , count_tuv in counts_tuv_dict.items():  # every individual tag and its count
        q[(t,u,v)] = count_tuv / counts_tu_dict[(t, u)]
    return q

def get_tags(counts):
    tags = []
    for key, value in counts.items():
        if (key[0] not in tags):
            tags.append(key[0])
    tags.append('STOP')
    # print(tags)
    return tags

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print ("Usage on Windows:  python transitionparams2.py [train file]")
        print ("Usage on Linux/Mac:  python3 transitionparams2.py [train file]")
        sys.exit()

    train_file = open(sys.argv[1], "r", encoding="utf-8")
    print ("Running selected train file :'{0}'".format(train_file))
    q,counts = estimateTransition(train_file)
    print(q)
    # for k,v in q.items():
        # print ("({0}, {1} -> {2}): {3}".format(k[0],k[1],k[2],v))
    print ("Run finish train file :'{0}'".format(sys.argv[1]))