import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
import openpyxl as xl
import cufflinks as cf


class LogNormalRandomWalk:

    def __init__(self):
        cf.set_config_file(offline=True)

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
                scaledRtns.append(((returns[i] - np.mean(returns)) / np.std(returns)) * (1/np.sqrt(timestep)))
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

    def empiricalPDF(self, scaledReturns):
        sampleBucketMid = self.getBucketMeans(scaledReturns)
        count = len(scaledReturns) - 1
        freq = np.histogram(scaledReturns, self.createBuckets(scaledReturns))
        frq = []
        for i in range(len(freq[0])):
            frq.append(freq[0][i])
        frq.append(0)

        empiricalPdf = []
        for i in range(len(sampleBucketMid)):
            empiricalPdf.append(frq[i] / count / self.getBucketWidth(scaledReturns))

        return empiricalPdf

    def normalPdf(self, scaledReturns):
        sampleBucketMid = self.getBucketMeans(scaledReturns)
        normalPdf = []
        for i in range(len(sampleBucketMid)):
            normalPdf.append(1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * sampleBucketMid[i] * sampleBucketMid[i]))

        return normalPdf

    def getBucketMeans(self, scaledReturns):
        sampleBucket = self.createBuckets(scaledReturns)
        bucketWidth = self.getBucketWidth(scaledReturns)
        sampleBucketMid = self.getBucketMids(sampleBucket, bucketWidth)
        return sampleBucketMid

    def getScaledReturns(self, timestep):
        returns = self.calculateReturn("../resources/Copy_JU241.1 SPX.xlsm", timestep)
        scaledReturns = self.scaleReturns(returns, timestep)
        return scaledReturns

    def combinePdfs(self):
        self.plotProperties()
        scaledReturns = self.getScaledReturns(1)
        self.plotPDFS(self.getBucketMeans(scaledReturns), self.empiricalPDF(scaledReturns))
        self.plotPDFS(self.getBucketMeans(scaledReturns), self.normalPdf(scaledReturns))
        plt.show()

    def qqplots(self):
        scaledReturns = self.getScaledReturns(2)
        stats.probplot(self.empiricalPDF(scaledReturns), plot=plt)
        sm.qqplot(self.empiricalPDF(scaledReturns), plot=plt)
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
