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

    # --- for first word (column) --- #
    if sentence[0] not in xs:
        sentence[0] = '#UNK#'

    for i in range(len(T)):    
        value = q.get((T[i],'START'),0) * e.get((sentence[0], T[i]), 0)   
        pi1[i][0] = ('START', value)
    # ----- for second word / col to k --- #
    for k in range(1, len(sentence)):
        if sentence[k] not in xs:
            sentence[k] = '#UNK#'
        word = sentence[k]

        for v in range(len(T)):
            temp = [] #array within node to be maxed
            for u in range(len(T)):
                value = pi1[u][k-1][1] * q.get((T[v], T[u]),0) * e.get((word, T[v]),0) 
                temp.append(value)
            max_value = max(temp)
            parent_node = temp.index(max_value) #index of parent node
            pi1[v][k] = (parent_node, max_value)

    # ---- Last Pi ---- #
    temp_last_pi = []
    for i in range(len(T)):    
        value = pi1[i][len(sentence)-1][1] * q.get(('STOP',T[i]),0) 
        temp_last_pi.append(value)
    max_value = max(temp_last_pi)
    parent_node = temp_last_pi.index(max_value)
    last_pi = (parent_node, max_value)

    # ------- backtracking --------#
    tags = []
    prev_node = last_pi[0]
    tags.append(T[prev_node])
    for i in range(1,len(sentence)):
        yn = pi1[prev_node][-i] #returns (index of node, probability)
        index = yn[0]
        tags.append(T[index])
        prev_node = index
    tags.reverse()
    return tags


# filename = "/Users/ganr/Desktop/ML/Project/EN/train"
dev_in = "/Users/ganr/Desktop/ML/Project/FR/dev.in"
output_file_name = "dev.p3.out"
q = estimateTransition(filename)

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
    


# # filesave_dir = os.path.join("/Users/ganr/Desktop/ML/Project/", "dev.p3.out")    
def create_test_result_file(test_result, filename):
    with open(filename, "w") as f:
        f.write(test_result)

result = generate_result(dev_in)
create_test_result_file(result, output_file_name)

