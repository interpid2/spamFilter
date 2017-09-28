#Pokrenut nltk.download() gdje je skinut
#punkt - tokenizer model
#stopwords corpus
#WordNet za WordNetLemmatizer

import matplotlib.pyplot as plt
import numpy as np
import re
from sklearn.metrics import confusion_matrix
from nltk.corpus import stopwords as sW
from nltk import word_tokenize as token
from nltk import WordNetLemmatizer as lemma
from collections import Counter as Cnt
import timeit as ti
from nltk import NaiveBayesClassifier as NBC
import os
import random as rnd

def customRead(dat):
    aList=[]
    aList=initLists(dat)
    allFeatures_custom=[getFeatures(email, True) for email in aList]
    result_custom=klasifikator.classify_many([fs for fs in allFeatures_custom])
    mailCnt=Cnt(result_custom).items()
    print 'Klasificirano ham mailova: ', mailCnt[0][1]
    print 'Klasificirano spam mailova: ', mailCnt[1][1]

def confMatrix(trueSet, predictedSet):
    c_matrix=confusion_matrix(trueSet, predictedSet)
    norm_conf = []
    
    for i in c_matrix:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.Greys, interpolation='nearest') 
 
    width = len(c_matrix)
    height = len(c_matrix[0]) 
 
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(c_matrix[x][y]), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center', color = 'green', size = 20)
    suma=np.sum(c_matrix)
    temp=np.array([float(c_matrix[0,0]+c_matrix[1,1])/suma,
                   1-float(c_matrix[0,0]+c_matrix[1,1])/suma,
                    float(c_matrix[0,0])/(np.sum(c_matrix[0])),
                    float(c_matrix[0,0])/(np.sum(c_matrix[:,0])),
                    float(c_matrix[1,1])/(np.sum(c_matrix[:,1]))
                   ])
    ParamEval=temp*100
    print "Tocnost: ", ParamEval[0], "%"
    print "Ucestalost pogresne klasifikacije: " , ParamEval[1], "%"
    print "Preciznost: ",  ParamEval[2], "%"
    print "Odziv: ",  ParamEval[3], "%"
    print "Specificnost: ",  ParamEval[4], "%"
    return ParamEval
    
def train(feat, tPer=0.7):
    trainSize=int(len(feat)*tPer)
    trainSet, testSet = feat[:trainSize], feat[trainSize:]
    infoTrain=Cnt(x[1] for x in trainSet).items()
    infoTest=Cnt(x[1] for x in testSet).items()
    print 'Velicina train seta (ham+spam=uk): ', infoTrain[0][1],"+",infoTrain[1][1],"=",len(trainSet)
    print 'Velicina test seta (ham+spam=uk): ', infoTest[0][1],"+",infoTest[1][1],"=", len(testSet)
    klasifikator = NBC.train(trainSet)
    return trainSet, testSet, klasifikator

def getFeatures(txt,mode=False):
    if mode==True:
        return dict({word: count for word, count in Cnt(emailPreprocess(txt)).items() if not word in stoplist})
    else:
        return {word: True for word in emailPreprocess(txt) if not word in stoplist}

def emailPreprocess(recenica):
    wn=lemma()
    return [wn.lemmatize(word.lower()) for word in token(recenica) if not re.match(r'^[0-9]+$',word)]        

def initLists(dat):
    aList=[]
    file_list=os.listdir(dat)
    for aFile in file_list:
        f=open(dat+aFile,'r')
        aList.append(f.read().decode('latin1'))
        f.close()
    return aList

start = ti.default_timer()
spam1 = initLists('enron1/spam/')
ham1 = initLists('enron1/ham/')
spam2 = initLists('enron2/spam/')
ham2 = initLists('enron2/ham/')

allEmails = [(email,'spam') for email in spam1]
allEmails += [(email,'ham') for email in ham1]
allEmails += [(email,'spam') for email in spam2]
allEmails += [(email,'ham') for email in ham2]
              

rnd.shuffle(allEmails)
stoplist=sW.words('english')
stoplist.append(u'cc')
allFeatures = [(getFeatures(email, True), label) for (email, label) in allEmails]

trainSet, testSet, klasifikator = train(allFeatures)
result=klasifikator.classify_many([fs for (fs, l) in testSet])
trueResult=[x[1] for x in testSet]
ParamEval=confMatrix(trueResult,result)
stop=ti.default_timer()
minute=int(stop-start)/60
sekunde=int(stop-start-(minute*60))
print 'Vrijeme obrade:',minute,"min ",sekunde,"sec"