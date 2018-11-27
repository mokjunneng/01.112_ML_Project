from part3 import *
from emissionparams import *

def viterbi(e,q, sentence):
    sentence = sentence.copy()
    result = []
    for i in zip(*e_dict.keys()):
        result.append(list((set(i))))

    xs = result[0] #the list of words
    T = result[1]
    T.append('START')
    T.append('STOP') #T is the array of tags with stop and start state
    
    ## --- Initialising array --- ##
    pi1 = [[0 for x in range(len(sentence))] for y in range(len(T))]  #x is number of words (col) #y is number of states/tags
    pi_index = [] #for backtracking

    # --- for first word (column) --- #
    if sentence[0] not in xs:
        sentence[0] = '#UNK#'

    for i in range(len(T)):    
        value = q.get((T[i],'START'),0) * e.get((sentence[0], T[i]), 0)   
        pi1[i][0] = value
    first_col = get_column(pi1,0)
    pi_index.append(first_col.index(max(first_col))) #save highest starting probability argmax()
    
    # ----- for second word / col to k --- #
    for k in range(1, len(sentence)):
        if sentence[k] not in xs:
            sentence[k] = '#UNK#'
        word = sentence[k]
        # print (word)

        node_max = [] #every state
        for v in range(len(T)):
            temp = [] #array within node to be maxed
            for u in range(len(T)):
                value = pi1[u][k-1] * q.get((T[v], T[u]),0) * e.get((word, T[v]),0) 
                temp.append(value)
                # print (T[v], T[u])
            max_value = max(temp)
            max_index = temp.index(max_value)
            pi1[v][k] = max_value
            node_max.append(max_value)
        pi_index.append(node_max.index(max(node_max)))

    # ---- Last Pi ---- #
    last_pi = []
    for i in range(len(T)):    
        value = pi1[i][len(sentence)-1] * q.get(('STOP',T[i]),0) 
        last_pi.append(value)
    pi_index.append(last_pi.index(max(last_pi))) #save highest starting probability

    # ------- backtracking --------#
    tags = []
    for i in range(0,len(sentence)):
        index = pi_index[-i]
        p = T[index]
        tags.append(p)
    tags.reverse()
    return tags
    # print(tags)

# filename = "/Users/ganr/Desktop/ML/Project/EN/train"
dev_in = "/Users/ganr/Desktop/ML/Project/EN/dev.in"
output_file_name = "dev.p3.out"
q = estimateTransition(filename)
test_sentence = ['NO', 'Saints', 'R', '.', 'Buch', 'might', 'come', 'back', 'n', 'play', 'vs', 'Seahawks', 'on', 'Sunday', '??']
tags = viterbi(e_dict, q, test_sentence)
print (tags)


def get_sentences(dev_in): #dev in is file
    #get sentence
    with open(dev_in, 'r') as f:
        dev = f.read().rstrip().split('\n\n')
    sentences = [] #array of each word in a sentence
    for i in range(len(dev)):
        sentences.append(dev[i].splitlines()) 
        
    return sentences


def generate_result(dev_in): #for the whole file
    test = get_sentences(dev_in) 
    # tags = []
    result = ""
    for sentence in test:
        tag_sentence = viterbi(e_dict, q, sentence) #tags for every sentence
        output_sentence = ""
        for i in range(len(sentence)):
            output_sentence += sentence[i] + " " + tag_sentence[i] + "\n"
        result += output_sentence + "\n"

    return result
    

# def create_test_result_file(test_result, filename):
#     with open(filename, "w") as f:
#         f.write(test_result)
#     print (filename)
# result = generate_result(dev_in)
# create_test_result_file(result, "dev.p3.out")


