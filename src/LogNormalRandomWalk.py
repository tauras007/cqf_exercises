import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import openpyxl as xl

class LogNormalRandomWalk:
    def readExcel(self, filename):
        return pd.read_excel(filename)

    def excelCol(self, fileName, colName):
        excel = self.readExcel(fileName)
        return excel.loc[:, colName]

    def calculateReturn(self, fileName, timestep):
        adjClose = self.excelCol(fileName, "Adj Close")
        rtns = []
        for i in range(len(adjClose)):
            if i < len(adjClose) - timestep:
                rtns.append((adjClose[i + timestep] - adjClose[i]) / adjClose[i])
        return rtns

    def scaleReturns(self, returns, timestep):
        scaledRtns = []
        for i in range(len(returns)):
            if i < len(returns):
                scaledRtns.append((returns[i] - np.mean(returns)) / np.std(returns))
        return scaledRtns

    def getBucketWidth(self, scaledRtns):
        min = np.min(scaledRtns)
        max = np.max(scaledRtns)
        bucketWidth = (max - min) / 200
        return bucketWidth

    def createBuckets(self, scaledRtns):
        bucketWidth = self.getBucketWidth(scaledRtns)
        sampleBucket = []
        for i in range(250):
            if i == 0:
                sampleBucket.append(np.min(scaledRtns))
            else:
                sampleBucket.append(sampleBucket[i - 1] + bucketWidth)
        return sampleBucket

    def getBucketMids(self, sampleBucket, bucketWidth):
        sampleBucketMid = []
        for i in range(len(sampleBucket)):
            sampleBucketMid.append(sampleBucket[i] - (0.5 * bucketWidth))
        return sampleBucketMid

    def empiricalPDF(self):
        returns = self.calculateReturn("../resources/Copy_JU241.1 SPX.xlsm", 2)
        scaledReturns = self.scaleReturns(returns, 2)
        sampleBucket = self.createBuckets(scaledReturns)
        bucketWidth = self.getBucketWidth(scaledReturns)
        sampleBucketMid = self.getBucketMids(sampleBucket, bucketWidth)
        count = len(scaledReturns) - 1
        freq = np.histogram(scaledReturns, sampleBucket)
        frq = []
        for i in range(len(freq[0])):
            frq.append(freq[0][i])
        frq.append(0)

        empiricalPdf = []
        for i in range(len(sampleBucketMid)):
            empiricalPdf.append(frq[i] / count / bucketWidth)

        self.plotPDFS(sampleBucketMid, empiricalPdf)

    def normalPdf(self):
        returns = self.calculateReturn("../resources/Copy_JU241.1 SPX.xlsm", 2)
        scaledReturns = self.scaleReturns(returns, 2)
        sampleBucket = self.createBuckets(scaledReturns)
        bucketWidth = self.getBucketWidth(scaledReturns)
        sampleBucketMid = self.getBucketMids(sampleBucket, bucketWidth)
        normalPdf = []
        for i in range(len(sampleBucketMid)):
            normalPdf.append(1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * sampleBucketMid[i] * sampleBucketMid[i]))

        self.plotPDFS(sampleBucketMid, normalPdf)

    def combinePdfs(self):
        self.plotProperties()
        self.empiricalPDF()
        self.normalPdf()
        plt.show()

    def plotProperties(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        plt.xlim(-4, 4)
        plt.ylim(0, 0.8)

    def plotPDFS(self, sampleBucketMid, pdf):
        plt.plot(sampleBucketMid, pdf)
