import numpy as np
import data
import neuralNetwork_template as nn
import logisticRegression_template as lr
import pandas as pd
import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt
#import pdb

visualPath = "visualization"

#評価値（交差エントロピー、正解率）のプロット
def plotEval(hDims, evalNN, evalLR, xlabel):
    #プロット
    plt.plot(hDims, evalNN, "o-", color="#0000FF")
    #レジェンド
    plt.legend(["Neural Network", "Logistic Regression"], fontsize=14)
    #各軸のラベル
    plt.xlabel("Number of hidden nodes", fontsize=14)
    plt.ylable(xlabel, fontsize=14)
    #表示範囲の設定
    max = np.max([np.max(evalNN), evalLR])
    plt.ylim(0, max+max*0.1)
    #ファイルに保存
    visuralPath="visualization"
    fullpath = os.path.join(visualPth,f"NN_LR_{xlabel}.png")
    plt.savefig(fullpath)

    plt.close()

#メインの始まり
if __name__ == "__main__":

    #1)データ生成
    #人工データの生成（難しい場合＋ノイズあり）
    myData = data.artificial(300,500,mean1=[1,2],mean2=[-2,-1],mean3=[4,-2],mean3multi=[-2,4], cov=[[1,0],[0,1]],noiseMean=[1,-2])
    batchSize = 0
    Nite = 1000
    learningRate = 0.01
    decayRate = 0.9999 #減衰率
    #2) 隠れ層のノード数の異なるニューラルネットワークモデルの作成
    classifierNN = []
    hDims = [1,5,10,50,100,200,300,500,800]
    for ind in np.arange(len(hDims)):
        classifierNN.append(nn.neuralNetwork(myData.xTrain, myData.tTrain, hDims[ind], batchSize=batchSize))
    #3) ロジスティックモデルの作成
    classifierLR = lr.logisticRegression(myData.xTrain, myData.tTrain, batchSize=batchSize)

    #4) モデルの学習
    for ite in np.arange(Nite):
        #評価の出力フラグ
        printEval = not ite % 100
        #ニューラルネットワークモデルの更新
        for ind in np.arange(len(hDims)):
            if printEval: print(f"Training NN{hDims[ind]} ite:{ite+1}", end='')
            classifierNN[ind].update(alpha=learningRate,printEval=printEval)
        #ロジスティックモデルの更新
        if printEval:print(f"Training LR ite{ite+1}", end='')
        classifierLR.update(alpha=learningRate, printEval=printEval)
        #5)更新幅の減衰
        learningRate *= decayRate

        if printEval:print("---------------")

    #6)評価
    #ニューラルネットワークの交差エントロピー損失の標準出力
    lossNN = []
    for ind in np.arange(len(hDims)):
        lossNN.append(classifierNN[ind].loss(myData.xTest, myData.tTest))
        print(f"Test loss NN{hDims[ind],lossNN[ind]}")
    #ロジスティックモデルの交差エントロピー損失の標準出力
    lossLR = classifierLR.loss(myData.xTest, myData.tTest)
    print(f"Test loss LR:{lossLR}")
    #交差エントロピー損失のプロット
    plotEval(hDims, lossNN, lossLR, "loss")
    #ニューラルネットワークのAccuracyの標準出力
    accuracyNN = []
    for ind in np.arange(len(hDims)):
        accuracyNN.append(classifierNN[ind], accuracy(myData.xTest, myData.tTest))
        print(f"Test accuracy NN{hDims[ind], accuracyNN[ind]}")
    #ロジスティックモデルのAccuracyの標準出力
    accuracyLR = classifierLR.accuracy(myData.xTest. myData.tTest)
    print(f"Test loss LR:{accuracyLR}")

    #Accuracyのプロット
    plotEval(hDims, accuracyNN, accuracyLR, "accuracy")
#メインの終わり