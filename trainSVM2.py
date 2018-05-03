from __future__ import print_function
from __future__ import division
import numpy as np
import skimage
from skimage import io,data
import glob
#import openface
#import cv2
#import dlib
#import csv
import scipy.io
from sklearn import svm, datasets
import sklearn.model_selection
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib as plt
from matplotlib import cm
from sklearn.externals import joblib
from scipy.cluster.vq import whiten,kmeans


def LoadData(basepath,typeOfFeat='dlib',person='gianni', typeofdata='testing'):
    ''' This function load the corresponding npy file with the features.
        Output is a table of features of a particular ID, of a particular
        set, obtained with a particular net'''
    
    alldata = np.load(basepath+typeofdata+"_"+person+"_feat.npy")
    if typeOfFeat == 'dlib':
        return alldata[0]
    else:
        return alldata[1]

def calcSelected(predProb,ths,othersId):
    ''' This is the probability based classifier. If an output is lower than a treshold,
        the classifier chooses for the <<others>> pseudo-class.  Arguments are:
        predProb = Prediction probabilities, a matrix with per class probabilities (on columns) of every image
        ths = threshold used for the decision
        othersId = index of the "others" class
        The output is an array of lenght = predProb.shape[0] with a decision for avery feature.'''
    selectedClass =  []
    for i in range(predProb.shape[0]): #cicla su ogni immagine
        maxId = np.argmax(predProb[i]) #trova il numero della classe a probabilità più alta
        if predProb[i][maxId] > ths: #se il valore di probabilità è maggiore della soglia
            selectedClass.append(maxId) #scegli per quella classe
        else:
            selectedClass.append(othersId) #altrimenti è "others"
    
    selectedClass = np.asarray(selectedClass)
    return selectedClass


def createBinaryDataSet(trainWho,negIdxTab,origSet,NSample):
    """ From  a multiclass set, this function creates a binary dataset.
        The function return a table of features and the corresponding labels.
        Arguments list:
        -Index of the class to be used as positive, (['gianni','stefano','sergio','jhilick','others'])
        -origin dataset,
        -samples per binary class,
        -useOthers as negative"""
    
    #positives        
    smallValidFeat = np.zeros((2*NSample,128))
    smallValidLabels = np.zeros(2*NSample)
    PosValidSetTemp = origSet[trainWho].copy()
    np.random.shuffle(PosValidSetTemp)
    smallValidFeat[0:NSample-1] = PosValidSetTemp[0:NSample-1]
    smallValidLabels[0:NSample-1] = 1

    #negatives
    indexNow = NSample
    negLen = len(negIdxTab)
    NegNSample = int(np.floor(NSample/negLen))
    for negIdx in range(negLen):
        NegSetTemp = origSet[int(negIdxTab[negIdx])].copy()
        np.random.shuffle(NegSetTemp)
        smallValidFeat[indexNow:indexNow+NegNSample-1] = NegSetTemp[0:NegNSample-1]
        smallValidLabels[indexNow:indexNow+NegNSample-1] = 0
        indexNow += NegNSample

    if indexNow < 2*NSample:
        diff = 2*NSample - indexNow
        smallValidFeat[indexNow:indexNow+diff-1] = NegSetTemp[indexNow:diff-1]
        smallValidLabels[indexNow:indexNow+diff-1] = 0
    return smallValidFeat,smallValidLabels

def CreatecolorTuple(f):
    ''' This function creates a colormap'''
    a = (1-f)/0.25;	#invert and group
    X = np.floor(a);	#this is the integer part
    Y = np.floor(255*(a-X)); #fractional part from 0 to 255
    if X == 0:
        r = 255
        g = Y
        b = 0
    elif X == 1:
        r = 255-Y
        g = 255
        b = 0
    elif X == 2:
        r = 0
        g = 255
        b = Y
    elif X == 3:
        r = 0
        g = 255-Y
        b = 255
    elif X == 4:
        r = 0
        g = 0
        b = 255
    
    r /= 255.0
    g /= 255.0
    b /= 255.0
    return (r,g,b)


######################## START

FeatTYPE = 'dlib' #openface o dlib
typeOfFeat = FeatTYPE
basepath = "/home/francesco/perLuca/sameImgs/" #folder feature condivise
basepathsavefig = '/home/francesco/test_svm/'

peopleName = ['gianni','stefano','sergio','jhilick','others']
folderName = ['testing','training','verification']
#testing_gianni_feat.npy

NsamplePerBClass = [25,50,100,200,400,800,1000] # how many samples for binary class
NTEST = 5 # how many classifiers train for a sampleNumber
TIMESVALID = 5 # how many validation for choosing a threshold
REPEATTEST = 5 # how many test on one trained classifier


#class to be trained
#trainWho = 3
avgAcc = np.zeros((len(peopleName)-1,len(NsamplePerBClass)))
avgVarAcc = np.zeros((len(peopleName)-1,len(NsamplePerBClass)))
#figallacc = plt.pyplot.figure(figsize=(16,10),dpi=400)

for trainWho in range(len(peopleName)-1):
    
    
    #we can use JET colormap
    #colorTuple = []
    #for i in range(len(NsamplePerBClass)):
    #    colorTuple.append(CreatecolorTuple(i/len(NsamplePerBClass)))
    #colorTuple = np.asarray(colorTuple)
    #print(colorTuple)

    #validation: 150 id, 150 others
    NSampleForValid = 150

    # test : 80 img ID, 80 img others
    NSampleForTest = 80

    testFeat = []
    trainFeat = []
    validFeat = []
    for name in peopleName:
        testFeat.append(LoadData(typeOfFeat=typeOfFeat,basepath=basepath,typeofdata='testing',person=name))
        trainFeat.append(LoadData(typeOfFeat=typeOfFeat,basepath=basepath,typeofdata='training',person=name))
        validFeat.append(LoadData(typeOfFeat=typeOfFeat,basepath=basepath,typeofdata='verification',person=name))

    print("name,#testing,#training,#verification")
    for nameidx in range(len(peopleName)):
        print(peopleName[nameidx],testFeat[nameidx].shape[0],trainFeat[nameidx].shape[0],validFeat[nameidx].shape[0])
    # all dataset loaded
    realNumberOfPeople = len(peopleName)-1

    fig = plt.pyplot.figure(figsize=(16,10),dpi=400)
    best_th = 0
    p = np.arange(0,1,0.01)
    testRes = np.zeros(len(NsamplePerBClass))
    varianceForClassTest = []
    for Nsidx in range(len(NsamplePerBClass)):
        print("Use ",NsamplePerBClass[Nsidx])
        nsampleNow = NsamplePerBClass[Nsidx]
        smallTrainFeat = np.zeros((2*nsampleNow,128)) # quanti devo caricare
        smallTrainLabel = np.zeros(2*nsampleNow)
        print(smallTrainFeat.shape)
        testcurve = np.zeros((NTEST,p.shape[0])) #accuracy curve per test
        for testN in range(NTEST): #test per classificatore
            negIdxTab = np.ones(len(peopleName))
            negIdxTab[trainWho] = 0
            negIdxTab[realNumberOfPeople] = 0 #others are ath the end
            idxr = np.arange(negIdxTab.shape[0])
            negIdxTab =(negIdxTab*idxr)[negIdxTab == 1]
            smallTrainFeat,smallTrainLabel = createBinaryDataSet(trainWho,negIdxTab,trainFeat,NSampleForValid)
            #done population of trainset
            #print('Done Trainset!')
            #print('Create svm model')
            #model = svm.SVC(kernel='linear', c=1, gamma=1)
            model = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma='auto', kernel='linear', #kernel='rgf' #overfit
                max_iter=-1, probability=True, random_state=None, shrinking=True,
                tol=0.001, verbose=True)
            #print('Fitting model')
            x_train = smallTrainFeat
            y_train = smallTrainLabel
            model.fit(x_train, y_train)
            score = model.score(x_train, y_train)
            #print("Model accuracy",score)
            #joblib.dump(model, "svm_dlib.pkl") # save SVM
            #print('Start validation')
            #create validation set
            veriaccuracy = np.zeros((TIMESVALID,p.shape[0]))


            for k in range(TIMESVALID): #repeat many times validation
                smallValidFeat, smallValidLabels = createBinaryDataSet(trainWho,[realNumberOfPeople],validFeat,NSampleForTest)
                #Validation set complete
                x_veri = smallValidFeat
                y_veri = smallValidLabels
                predicted_proba = model.predict_proba(x_veri)

                boolmat = np.zeros((p.size,y_veri.shape[0]))
                eqlabel = np.zeros_like(p)
                correctpred = np.zeros_like(p)
                for j in range(0,p.size):
                    boolmat[j] = calcSelected(predicted_proba,p[j],0)
                    eqlabel[j] = np.sum(np.equal(y_veri,boolmat[j]))
                    correctpred[j] = eqlabel[j]/y_veri.size
                veriaccuracy[k] = correctpred.copy() # accuracy curve for this validation

            meanveriFun = np.mean(veriaccuracy,axis=0)
            best_th = np.argmax(meanveriFun) # manually selected
            choosedTh =  p[best_th]
            print("Choosed threshold:",choosedTh)

            #plot validation
            #print("Threshold choosed at:", choosedTh)
            #plt.pyplot.figure()
            #plt.pyplot.xlabel("Threshold")
            #plt.pyplot.ylabel("accuracy")
            #plt.pyplot.title(FeatTYPE + " - Acc = Combined accuracy(threshold)")
            #plt.pyplot.plot(p,meanveriFun)
            #plt.pyplot.grid(True)
            #plt.pyplot.show()

            #start test
            print('Testing')
            alltest = np.zeros((REPEATTEST,p.shape[0]))
            for jj in range(REPEATTEST):
                x_test,y_test = createBinaryDataSet(trainWho,[realNumberOfPeople],testFeat,NSampleForTest)
                predicted_proba = model.predict_proba(x_test)
                boolmat = np.zeros((p.size,y_test.shape[0]))
                eqlabel = np.zeros_like(p)
                correctpred = np.zeros_like(p)
                for j in range(0,p.size):
                    boolmat[j] = calcSelected(predicted_proba,p[j],0)
                    eqlabel[j] = np.sum(np.equal(y_test,boolmat[j]))
                    correctpred[j] = eqlabel[j]/y_test.size
                alltest[jj] = correctpred.copy()  
            testcurve[testN] = np.mean(alltest,axis=0).copy() #media dei test su tanti samples per un classificatore

        meanTestFun = np.mean(testcurve,axis=0) # media su tanti classificatori
        testRes[Nsidx] = meanTestFun[best_th] # sui classificatori accuratezza media con quella soglia
        # plotta statistiche per quel numero di sample
        plt.pyplot.plot(p,meanTestFun,label=str(nsampleNow),color=cm.jet(Nsidx/len(NsamplePerBClass)))
        plt.pyplot.scatter(choosedTh,testRes[Nsidx],color=cm.jet(Nsidx/len(NsamplePerBClass)))
        plt.pyplot.text(choosedTh,testRes[Nsidx],str("%.2f"%testRes[Nsidx]),color=cm.jet(Nsidx/len(NsamplePerBClass)))
        varNow = np.std(testcurve,axis=0)[best_th]
        varianceForClassTest.append(varNow)
        plt.pyplot.errorbar(p[best_th],testRes[Nsidx],yerr=varNow,fmt='o',color=cm.jet(Nsidx/len(NsamplePerBClass)))
        print(testRes)




    plt.pyplot.title("Test accuracy(threshold), - " + FeatTYPE + "-  class: "+str(trainWho) + "_" + peopleName[trainWho],fontsize=30)
    plt.pyplot.xlabel("Threshold",fontsize=25)
    plt.pyplot.ylabel("Accuracy",fontsize=25)
    axes = fig.gca()
    axes.set_ylim([0.8,1])
    plt.pyplot.grid(True)
    #plt.pyplot.axvline(choosedTh,linewidth=3)
    plt.pyplot.legend()
    plt.pyplot.show()
    fig.savefig(basepathsavefig +FeatTYPE+"_"+str(trainWho)+"_"+peopleName[trainWho]+"accuracy_p_" + '.png', bbox_inches='tight',dpi=300)

    fig = plt.pyplot.figure(figsize=(16,10),dpi=400)
    plt.pyplot.plot(np.asarray(NsamplePerBClass),testRes)
    plt.pyplot.errorbar(np.asarray(NsamplePerBClass),testRes,yerr=np.asarray(varianceForClassTest),fmt='o')
    #plt.pyplot.text(np.asarray(NsamplePerBClass),testRes,str("%.2f"%testRes[Nsidx]))
    plt.pyplot.title('Accuracy(Samples per class) -' +  str(trainWho) +' ' + peopleName[trainWho] + ' ' + FeatTYPE,fontsize=30)
    plt.pyplot.xlabel('Samples per binary class during training',fontsize=25)
    plt.pyplot.ylabel("Accuracy",fontsize=25)
    axes = fig.gca()
    axes.set_ylim([0.8,1])
    plt.pyplot.grid(True)
    plt.pyplot.show()
    fig.savefig(basepathsavefig +FeatTYPE+ "_"+ str(trainWho)+"_"+peopleName[trainWho]+" accuracy_p_N" + '.png', bbox_inches='tight',dpi=300)
    #np.save(basepathsavefig + FeatType + + "_" + str(trainWho)+"_"+peopleName[trainWho]+ "accN",testRes)
    #np.save(basepathsavefig + FeatType + + "_" + str(trainWho)+"_"+peopleName[trainWho]+ "acc_VarN",np.asarray(varianceForClassTest))
    
    avgAcc[trainWho] = testRes.copy()
    avgVarAcc[trainWho] = np.asarray(varianceForClassTest)


avgAccfinal = np.mean(avgAcc,axis=0)
avgVarAccfinal = np.mean(avgVarAcc,axis=0)
print(avgAcc)
print(avgVarAccfinal)

fig = plt.pyplot.figure(figsize=(16,10),dpi=400)
plt.pyplot.plot(np.asarray(NsamplePerBClass),avgAccfinal)
plt.pyplot.errorbar(np.asarray(NsamplePerBClass),avgAccfinal,yerr=avgVarAccfinal,fmt='o')

plt.pyplot.title('Average Accuracy(Samples per class) - ' + FeatTYPE,fontsize=25)
plt.pyplot.xlabel('Samples per binary class during training',fontsize=20)
plt.pyplot.ylabel("Accuracy",fontsize=20)
axes = fig.gca()
#axes.set_ylim([0.8,1])
plt.pyplot.grid(True)
plt.pyplot.show()
fig.savefig(basepathsavefig +FeatTYPE+ "_"+ " average accuracy_p_N" + '.png', bbox_inches='tight',dpi=300)
print('All done')    
