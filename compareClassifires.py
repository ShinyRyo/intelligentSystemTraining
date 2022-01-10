import numpy as np
import data
import nuralNetwork_template as nn
import LogisticRegression_template as lr
import pandas as pd
import pickle
from datetime import datetime
import os
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