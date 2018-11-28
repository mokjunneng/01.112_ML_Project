#part 4a
import sys

def estimateTransition(train_file):
    
    with open(train_file, 'r', encoding="utf-8") as f:
        data = f.read().rstrip().splitlines() #removes empty array at the end and splits each line into an array

    counts_tuv_dict = {} #count transition from t,u --> v
    counts_tu_dict = {} #number of occurence for each tag pair (t,u)
    
    for i in range(len(data)-2):
        # print ("Reading line: {0}".format(i))
        line_t = data[i]
        line_u = data[i+1]
        line_v = data[i+2]

        # if t,u lines not empty 
        if (len(line_t) != 0 and len(line_u) != 0): 
            word_t, tag_t = line_t.split(" ")[0],line_t.split(" ")[-1]
            word_u, tag_u = line_u.split(" ")[0],line_u.split(" ")[-1]
            # if v line not empty <t,u,v>
            if (len(line_v) != 0): 
                word_v, tag_v = line_v.split(" ")[0],line_v.split(" ")[-1]
            else: # <t,u,stop>
                word_v, tag_v = [None,'STOP']
              
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

        # if u line is empty <t,stop,other>, do not do anything

        # if t line is empty <start,u,v>
        elif (len(line_t) == 0):
            word_t, tag_t = [None,'START']
            if (len(line_u) != 0):
                word_u, tag_u = line_u.split(" ")[0],line_u.split(" ")[-1]
            else:
                continue # double blank line

            if (len(line_v) != 0):
                word_v, tag_v = line_v.split(" ")[0],line_v.split(" ")[-1]
            else: # <start,u,STOP>
                word_v, tag_v = [None,'STOP']

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

    q = {}
    #for every state: y_i-2 = t, y_i-1 = u, y_i = v
    # i-2,i-1 -> i 
    for (tuv_t, tuv_u, tuv_v) , count_tuv in counts_tuv_dict.items():  # every individual tag and its count
        q[(tuv_t, tuv_u, tuv_v)] = count_tuv / counts_tu_dict[(tuv_t, tuv_u)]
    return q

if __name__ == "__main__":
    if len(sys.argv) > 2:
        print ("Usage on Windows:  python transitionparams2.py [train file]")
        print ("Usage on Linux/Mac:  python3 transitionparams2.py [train file]")
        sys.exit()
        
    elif len(sys.argv) == 1:
        train_file = "EN/train"
        print ("Running default train file :'{0}'".format(train_file))
        out_dict = estimateTransition(train_file)
        for k,v in out_dict.items():
            print ("({0}, {1} -> {2}): {3}".format(k[0],k[1],k[2],v))
        print ("Run finish train file :'{0}'".format(train_file))

    elif len(sys.argv) == 2:
        train_file = sys.argv[1]
        print ("Running selected train file :'{0}'".format(train_file))
        out_dict = estimateTransition(train_file)
        for k,v in out_dict.items():
            print ("({0}, {1} -> {2}): {3}".format(k[0],k[1],k[2],v))
        print ("Run finish train file :'{0}'".format(train_file))