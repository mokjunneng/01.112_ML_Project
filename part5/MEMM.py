import nltk
from nltk.classify import MaxentClassifier
import sys, os
import re

#global variables



change_of_sentence_flag = 0 #a marker for the end of sentence
dicE = {} #temporarry dic 
wordStartList = []
labeled_features = [] # a list of (word, tag, previous_tag)
tag_unique_list = []
tag_full_list = []
tag_end_list = []

def main(train_file, test_file):
    global change_of_sentence_flag
    wordList = [] # store words in a sentence
    tagList = [] # store tags in a sentence

    # read file and fill in wordStartList & tagList
    read_train_file(train_file)

    calc_end()

    # train a MaxEnt classifier based on train set
    maxent_classifier = train_classifier()

    # read test file
    for line in test_file:
        if line.strip(): # if not empty do following
            sentenceList = line.split()
            word = sentenceList[0]
            tag = sentenceList[-1]
            wordList.append(word)
            tagList.append(tag)
            # store words that are begining of the sentence
            if change_of_sentence_flag == 1:
                wordStartList.append(word)
                change_of_sentence_flag = 0
        if not line.strip(): # empty line
            change_of_sentence_flag = 1
            previous_tag = 'START'
            path = MEMM(wordList, tagList, maxent_classifier) # list of tags returned by HMM function call
            print ("path: {0}".format(path))
            
            for i in range(len(wordList)): #part_of_speech_tag(tagList) and token_list(wordList) has the same length
                output_file.write(wordList[i]+b" "+ path[i]+ b"\n")

            output_file.write(b"\n")
            wordList = [] # refresh word list
            tagList = [] # clear list
            boiList = [] # clear list
            # prob_table = {}#refresh prob_table

def MEMM(wordList,tagList,maxent_classifier):
	w1 = wordList[0] # the first word of the sentence
	t1 = tagList[0] # the first tag of the sentence
	tRange = len(tag_unique_list)
	wRange = len(wordList)

	viterbi = [[0 for x in range(len(wordList)+2)] for x in range(len(tag_unique_list)+2)] 
	backpointer = [['' for x in range(len(wordList)+2)] for x in range(len(tag_unique_list)+2)] 
	#============ intialization ========================
	for t in range(tRange): # loop through every tag
		probability = maxent_classifier.prob_classify(MEMM_features(w1,t1, 'START' )) 
		posterior = float(probability.prob(tag_unique_list[t]))
		# score transition 0(start) -> q given w1
		viterbi[t][1] = posterior
		backpointer[t][1] = 0 # stand for q0 (start point)

	#============ for word w from 2 to T ===============
	maxViterbi = 0
	maxPreviousState = 0 
	maxPreTerminalProb = 0
	for w in range (1, wRange):	
		for t in range (tRange):
			#find max verterbi = max (previous * posterior)	
			word = wordList[w]
			tag = tagList[w]
			probability = maxent_classifier.prob_classify(MEMM_features(word,tag,tag_unique_list[0])) 
			posterior = float(probability.prob(tag_unique_list[t]))
			maxViterbi = float(viterbi[0][w]) * posterior
			maxPreviousState = 0
			for i in range (1, tRange):
				word = wordList[w]
				tag = tagList[w]
				probability = maxent_classifier.prob_classify(MEMM_features(word,tag,tag_unique_list[i])) 
				posterior = float(probability.prob(tag_unique_list[t]))
				if float(viterbi[i][w]) * posterior > maxViterbi:
					 maxViterbi = float(viterbi[i][w]) * posterior
					 maxPreviousState = i # content tag_unique_list[i]		
			viterbi[t][w+1] = maxViterbi	
			backpointer[t][w+1] = tag_unique_list[maxPreviousState] # points to the matrix x axis (max previous)
			
			maxViterbi = 0
			maxPreviousState = 0
			maxPreTerminalProb = 0
	#termination step
	#viterbi[qF, T] = max (viterbi[s,T] *as,qF)
	maxPreTerminalProb = float(viterbi[0][wRange-1] )* float(dicE[tag_unique_list[0]]['STOP'])
	
	maxPreviousState = 0
	for i in range (1, tRange):
		if float(viterbi[i][wRange]) * float(dicE[tag_unique_list[i]]['STOP']) > maxPreTerminalProb:
			maxPreTerminalProb = float(viterbi[i][wRange]) * float(dicE[tag_unique_list[i]]['STOP']) 

			maxPreviousState = i
		
			#print ("maxPreTerminalProb: " + str(maxPreTerminalProb))
	viterbi[tRange][wRange+1] = maxPreTerminalProb 
	backpointer[tRange][wRange+1] = tag_unique_list[maxPreviousState]
	#return POS tag path 
	pathReverse = [tag_unique_list[maxPreviousState]]
	maxPreviousTag = tag_unique_list[maxPreviousState]
	
	i = 0
	while i < (wRange -1):
		pathReverse.append(backpointer[tag_unique_list.index(maxPreviousTag)][wRange - i])
		maxPreviousTag = backpointer[tag_unique_list.index(maxPreviousTag)][wRange - i]
		i = i + 1 

	#reverse the path to make it correct
	index = len(pathReverse)
	path = []
	while index >= 1 :
		path.append(pathReverse[index - 1])
		index = index -1 
	return path


def read_train_file(input_file):
    """ Read file and fill in wordStartList, tag_unique_list, tag_full_list, labeled_features """
    previous_tag = 'START'

    change_of_sentence_flag = 0 #a marker for the end of sentence

    for line in input_file:
        if not line.strip(): # empty line
            change_of_sentence_flag = 1
            previous_tag = 'START'
        else: 
            sentenceList = line.split()
            word = sentenceList[0]
            tag = sentenceList[-1]
            # print(word)

            #store tag_unique_list
            if (tag not in tag_unique_list):
                tag_unique_list.append(tag)

            #store words that are begining of the sentence
            if change_of_sentence_flag == 1:
                wordStartList.append(word)
                tag_end_list.append(tag_full_list[-1])
                change_of_sentence_flag = 0

            tag_full_list.append(tag)
            item = word, tag, previous_tag
            labeled_features.append(item)
            previous_tag = tag
    # print ("wordStartList: {0}".format(wordStartList))
    # print ("labeled_features: {0}".format(labeled_features))
    # print ("tag_unique_list: {0}".format(tag_unique_list))
    # print ("tag_full_list: {0}".format(tag_full_list))
    # print ("tag_end_list: {0}".format(tag_end_list))
    input_file.close()

def calc_end():
    # calculate the End prior
    global dicE
    countTag = 0
    countEnd = 0
    # calculate the prior (End|state) = C(state, End)/C(state) 
    for i in tag_unique_list:
        for j  in range(len(tag_end_list)):
            for f in tag_full_list:
                if j == 0:
                    if i == f:
                        countTag = countTag + 1 
            if i == tag_end_list[j]:
                countEnd = countEnd + 1 
        ProbE = format(countEnd/(countTag*1.0), '.5f')
        dicE.update({i: {'STOP':ProbE}})

        countEnd = 0
        countTag = 0

def MEMM_features(word, tag, previous_tag):
    """ Return a dictionary contains input features for each (word, tag, previous_tag) """
    features = {}
    features['current_word'] = word
    features['current_tag'] = tag
    if str(word[0]).isalpha():
        features['capitalization'] = word[0].isupper()
    else:
        features['capitalization'] = False
    features['start_of_sentence'] = word in wordStartList
    if str(word[0]).isalpha():
        features['cap_start'] = word not in wordStartList and word[0].isupper()
    else:
        features['cap_start'] = False
    features['previous_NC'] = previous_tag

    return features


def train_classifier():
    """ Train a MaxEnt classifier and return it """
    labeled_featuresets = [(MEMM_features(word, tag, previous_tag), tag) for (word, tag, previous_tag) in labeled_features]
    maxent_classifier = MaxentClassifier.train(labeled_featuresets, max_iter=50)
    return maxent_classifier

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print ('Please make sure you have installed Python 3.4 or above!')
        print ("Usage on Windows:  python emission.py [train file] [dev.in file]")
        print ("Usage on Linux/Mac:  python3 emission.py [train file] [dev.in file]")
        sys.exit()


    path = os.path.dirname(sys.argv[1])

    training_file = open(sys.argv[1], "rb")
    testing_file = open(sys.argv[2], "rb")
    output_file = open("{0}/dev.p5.out".format(path), "wb")

    main(training_file, testing_file)