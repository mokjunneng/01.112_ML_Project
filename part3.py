#part 3a
def estimateTransition(train):
    
    #train is file name
    with open(train, 'r') as f:
        data = f.read().rstrip().splitlines() #removes empty array at the end and splits each line into an array
 
    counts = {} #number of occurence for each tag (u)
    counts_uv = {} #count transition from u --> v
    count_start = 1 #alternative way to calc the number of start / end tag
    
    
    for i in range(len(data)):
        element = data[i]
        #counting number of occurence for each tag and transition
        if (len(element) != 0): #empty arrays are not parsed
            temp = element.split()
            # count_start += 1
            if (temp[1] not in counts):
                counts[temp[1]] = 1
            else:
                counts[temp[1]] += 1

            u = temp[1] #current tag
            
            if (i != len(data)-1): #check if its the last line of the whole data
                if (len(data[i+1])!= 0): #check if next tag is empty
                    v = data[i+1].split()[-1] #next tag
                    uv = (u,v)
                    if (uv not in counts_uv):
                        counts_uv[uv] = 1
                    else:
                        counts_uv[uv] += 1
        
              
        elif (len(element) == 0):
            start_y1 = ('START', data[i+1].split()[-1])
            
            count_start += 1
            if (start_y1 not in counts_uv):
                counts_uv[start_y1] = 1
            else:
                counts_uv[start_y1] += 1

            stop_yn = (data[i-1].split()[-1], 'STOP')
            
            if (stop_yn not in counts_uv):
                counts_uv[stop_yn] = 1
            else:
                counts_uv[stop_yn] += 1

    counts['START'] = count_start
    counts['STOP'] = count_start
    q = {}
    #for every state: y_j = u, y_i = v, j current, i for next
    # j --> i 
    for y_j , count_y in counts.items(): #every individual tag and its count
        for y_i in counts:
            # print (y_i, y_j)
            if ((y_j, y_i) in counts_uv):
                q[(y_i, y_j)] = counts_uv.get((y_j, y_i)) / count_y
    # print (q)    
    return q