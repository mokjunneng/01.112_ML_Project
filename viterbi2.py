from transitionparams2 import *
from emissionparams import *
import os
import math

def viterbi(e,q, sentence): # 2nd-order
    sentence = sentence.copy()
    result = []
    for i in zip(*e_dict.keys()):
        result.append(list((set(i))))

    xs = result[0] #the list of words
    T = result[1]
    T.append('START')
    T.append('STOP') #T is the array of tags with stop and start state
    
    ## --- Initialising 3d array: pi1[current node y1][previous node y2][current node x] --- ##
    pi1 = [[[0 for x in range(len(sentence))] for y2 in range(len(T))] for y1 in range(len(T))]  #x is number of words (col) #y is number of states/tags
    # --- for second word (column) --- #
    if sentence[0] not in xs:
        sentence[0] = '#UNK#'

    for i in range(len(T)): # i current
        for j in range(len(T)): # j previous
            value = q.get(('START',T[j],T[i]),0) * e.get((sentence[0], T[i]), 0) *10000
            try:
                value = math.log(value)
            except:
                value = 0.00001
                value = math.log(value)
            # print (value)
            pi1[i][j][1] = (('START',j), value)

    # ----- for third word / col to k --- #
    for k in range(2, len(sentence)):
        if sentence[k] not in xs:
            sentence[k] = '#UNK#'
        word = sentence[k]

        for v in range(len(T)): # v current node
            temp = [[0.0] for x in range(len(T))] # array within node to be maxed
            for u in range(len(T)): # u previous node
                for t in range(len(T)): # t pre-previous node
                    value = pi1[u][t][k-1][1] * q.get((T[t], T[u], T[v]),0) * e.get((word, T[v]),0)*10000
                    try:
                        value = math.log(value)
                    except:
                        value = 0.00001
                        value = math.log(value)
                    # print (value)
                    temp[u].append(value)
                max_value = max(max(temp))
                parent_u, parent_t = index2d(temp,max_value) # index of parent node
                pi1[v][u][k] = ((parent_t, parent_u), max_value) # only 1 max parent pair for each current node
    # ---- Last Pi ---- #
    temp_last_pi = [[0 for x in range(len(T))] for y in range(len(T))]
    for i in range(len(T)): # i pre
        for j in range(len(T)): # j pre-previous
            
            value = pi1[i][j][len(sentence)-1][1] * q.get((T[i],T[j],'STOP'),0)*10000
            try:
                value = math.log(value)
            except:
                value = 0.00001
                value = math.log(value)
            # print (value)
            temp_last_pi[i][j] = value

    max_value = max(max(temp_last_pi))
    parent_u, parent_t = index2d(temp_last_pi,max_value)
    last_pi = ((parent_t, parent_u), max_value) # pi for 'STOP' node
    # ------- backtracking --------#
    print ("---------------backtracking----------------")
    tags = []
    prev_prev_node, prev_node = last_pi[0]
    tags.append(T[prev_node])
    tags.append(T[prev_prev_node])
    for k in range(1,len(sentence)-1):
        yn = pi1[prev_node][prev_prev_node][-k] # returns (index of node, probability)
        index = yn[0] # (t,u)
        tags.append(T[index[1]]) # only track to previous node u (not preprevious t)
        prev_prev_node, prev_node = index
    tags.reverse()
    # print("tags: {0}".format(tags))
    return tags

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
    

def create_test_result_file(test_result, filename):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(test_result)

def index2d(list2d, value):
    return next((i, j) for i, lst in enumerate(list2d) 
                for j, x in enumerate(lst) if x == value)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("Usage on Windows:  python viterbi2.py [train file] [dev.in file]")
        print ("Usage on Linux/Mac:  python3 viterbi2.py [train file] [dev.in file]")
        sys.exit()

    q = estimateTransition(sys.argv[1])
    e_dict = train(sys.argv[1])
    result = generate_result(sys.argv[2])
    path = os.path.dirname(sys.argv[1])

    create_test_result_file(result, "{0}/dev.p4.out".format(path))