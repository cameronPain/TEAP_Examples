#!/usr/bin/env python3
#
# Cameron Pain (cameron.pain@austin.org.au) Run a previously constructed model for training, evaluating or predicting.
#
# cdp 20191003
#
import numpy as n
import matplotlib.pyplot as pyplot
import os
import time
import sys
workingDir = os.popen('pwd').read().split('\n')[0]
sys.path.append(workingDir + '/Imports/')
#Import tensorflow stuff

#Import tensorflow stuff
import tensorflow as tf
from   tensorflow.keras.models import Sequential
from   tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
#Import my own modules
import NeuNetDataParser as NNDP


def main(model, Data_Dir, addRandomTranslations, normaliseData, epochs, batch_size, saveModel, dontSave, checkInputData):
    print('Starting.')
    start = time.time()
    MODEL = tf.keras.models.load_model(model)
    data, label = NNDP.getTrainingEvaluationData(Data_Dir + '/data', Data_Dir + '/label', addRandomTranslations = addRandomTranslations, normaliseData = normaliseData )
    print('Data shapes: ')
    print('     training Im: ' + str(n.shape(data)))
    print('     training L : ' + str(n.shape(label)))
    if checkInputData:
        for i in range(len(data)):
            f = pyplot.figure()
            a = f.add_subplot(1,2,1)
            b = f.add_subplot(1,2,2)
            a.imshow(data[i,:,:,0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = n.amax(data[i,:,:,0]))
            b.imshow(label[i,:,:,0], cmap = pyplot.cm.binary, vmin = 0.0, vmax = 1)
            a.set_xlabel('Input Data')
            b.set_xlabel('Ground Truth Binary Mask')
            pyplot.show()
    print('Training model.')
    output_channels = n.shape(data[:-1])
    optimiser = 'adam'
    loss      = 'binary_crossentropy'
    metrics   = ['accuracy']
    MODEL.compile(optimizer = optimiser, loss = loss, metrics = metrics)
    MODEL.fit(data, label, epochs = epochs, batch_size = batch_size)
    print('Model trained.')
    if dontSave == False:
        MODEL.save(saveModel)
        print('Model saved as ' + saveModel)
    else:
        print('Model not saved.')
    end = time.time()
    print('Time elapsed: ' + str(end - start) + ' seconds')
    print('\n\n\n\n')
    sys.exit()


import argparse
if __name__ == '__main__' :
    usage = 'Cameron Pain (cameron.pain@austin.org.au) Run a previously constructed model for training, evaluating or predicting.'
    parser = argparse.ArgumentParser(description = usage)
    #positional
    parser.add_argument('model'   ,   type = str  , help = 'Path to the saved model. If you are running in train mode, you will ovewrite the saved file.')
    parser.add_argument('Data_Dir',   type = str  , help = 'Path to the directory containing your data. If you are in trainMode or evalMode, the data directory should contain two subdirectories named "data" and "label" containing the appropriate training data and labels. Note that the file names of a training datum and its corresponding label need to be the same. In predict mode, the directory should contain only the data file you wish to run.')
    #Mode
    parser.add_argument('--addRandomTranslations',  dest = 'addRandomTranslations', default = False, action = 'store_true', help = 'Set this flag to include 10 random translation images for each input datum.')
    parser.add_argument('--normaliseData',          dest = 'normaliseData'        , default = False, action = 'store_true', help = 'Set this flag to normalise the input data. Note if you trained the model with normalised data, you will want to normalise your prediction and evaluation data.')
    parser.add_argument('--epochs',                 type = int                    , default = 10                          , help = 'The number of epochs to train for.')
    parser.add_argument('--batch_size',             type = int                    , default = 10                          , help = 'The batch size to use when training.')
    saveGroup = parser.add_mutually_exclusive_group(required = True)
    saveGroup.add_argument('--saveModel'     ,              type = str                    , default = 'Model_' + str(time.time()) + '.model'  , help = 'The path and name of the saved .model file. If you put just a file name, it will save into the directory you are operating in. The default save is in the current directory and saved as model with the unix time.')
    saveGroup.add_argument('--dontSave'      , dest = 'dontSave', default = False, action = 'store_true',  help = 'Set this flag to not save an output.')
    #Evaluate Mode
    parser.add_argument('--checkInputData', dest = 'checkInputData', default = False, action = 'store_true', help = 'Set this flag to show the input data and the corresponding label.')
    args = parser.parse_args()
    main(args.model, args.Data_Dir, args.addRandomTranslations,  args.normaliseData, args.epochs, args.batch_size, args.saveModel, args.dontSave, args.checkInputData)



#end if