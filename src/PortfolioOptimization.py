import pandas as pd
import scipy.optimize as opt
import numpy as np
import yfinance as yf
from numpy.linalg import multi_dot
import plotly.graph_objects as go


class PortfolioOptimization:

    def __init__(self):
        self.tradingDays = 252

    def getData(self, stocks, start, end):
        stockData = yf.download(stocks, start, end, progress=False)['Adj Close']
        returns = stockData.pct_change().dropna()
        standardDev = returns.std()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix, standardDev

    def portfolioVariance(self, weights, meanReturns, covMatrix):
        return self.portfolioPerformance(weights, meanReturns, covMatrix)[1]

    def portfolioPerformance(self, weights, meanReturns, covMatrix):
        annualReturns = np.sum(weights * meanReturns) * self.tradingDays
        annualStdDeviation = np.sqrt(multi_dot([weights.T, covMatrix, weights])) * np.sqrt(self.tradingDays)
        return annualReturns, annualStdDeviation

    def negativeSharpeRatio(self, weights, meanReturns, covMatrix, riskFreeRate):
        pReturns, pStd = self.portfolioPerformance(weights, meanReturns, covMatrix)
        return - (pReturns - riskFreeRate) / pStd

    def maxSharpeRatio(self, meanReturns, covMatrix, riskFreeRate, constrainSet=(0, 1)):
        """Minimize the negative sharpe ratio, by altering the weights of the portfolio"""
        numAssets = len(meanReturns)

        args = (meanReturns, covMatrix, riskFreeRate)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constrainSet
        bounds = tuple(bound for asset in range(numAssets))
        result = opt.minimize(self.negativeSharpeRatio, numAssets * [1. / numAssets], args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        return result

    def minimizeVariance(self, meanReturns, covMatrix, constrainSet=(0, 1)):
        """Minimize the portfolio variance by altering the weights/allocation of assets in the portfolio"""
        numAssets = len(meanReturns)
        args = (meanReturns, covMatrix)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constrainSet
        bounds = tuple(bound for asset in range(numAssets))
        result = opt.minimize(self.portfolioVariance, numAssets * [1. / numAssets], args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)
        return result

    def calculatedResults(self, meanReturns, covMatrix, riskFreeRate, constrainSet=(0, 1)):
        """Read mean, covariance and other financial information, output Max Sharpe Ratio and Minimized Variance and
        finally the efficient frontier"""
        """Max Vol portfolio"""
        maxSR_Portfolio = self.maxSharpeRatio(meanReturns, covMatrix, riskFreeRate)
        maxSR_ret, maxSR_std = self.portfolioPerformance(maxSR_Portfolio['x'], meanReturns, covMatrix)
        maxSrAllocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
        maxSrAllocation.allocation = [round(i * 100, 2) for i in maxSrAllocation.allocation]

        """Min Vol Portfolio"""
        minVar_Portfolio = self.minimizeVariance(meanReturns, covMatrix)
        minVar_ret, minVar_std = self.portfolioPerformance(minVar_Portfolio['x'], meanReturns, covMatrix)

        minVarAllocation = pd.DataFrame(minVar_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
        minVarAllocation.allocation = [round(i * 100, 2) for i in minVarAllocation.allocation]

        """Efficient Frontier List"""
        efficientFrontierList = []
        targetReturns = np.linspace(minVar_ret, maxSR_ret, 20)
        for target in targetReturns:
            efficientFrontierList.append(self.efficientOptimization(meanReturns, covMatrix, target)['fun'])

        maxSR_ret, maxSR_std = round(maxSR_ret * 100, 2), round(maxSR_std * 100, 2)
        minVar_ret, minVar_std = round(minVar_ret * 100, 2), round(minVar_std * 100, 2)

        return maxSR_ret, maxSR_std, maxSrAllocation, minVar_ret, minVar_std, minVarAllocation, efficientFrontierList, targetReturns

    def EE_Graph(self, meanReturns, covMatrix, riskFreeRate, constrainSet=(0, 1)):
        """Graph the min vol, max sharpe ratio and efficient frontier"""
        maxSR_ret, maxSR_std, maxSrAllocation, minVar_ret, minVar_std, minVarAllocation, efficientFrontierList, targetReturns = (
            self.calculatedResults(meanReturns, covMatrix, riskFreeRate, constrainSet))

        rfr = go.Scatter(name='Maximum Sharpe Ratio',
                                    mode='lines+markers',

                                    y=[riskFreeRate],
                                    marker=dict(color='blue', size=14, line=dict(width=3, color='black')))

        maxSharpeRatio = go.Scatter(name='Maximum Sharpe Ratio',
                                    mode='markers',
                                    x=[maxSR_std],
                                    y=[maxSR_ret],
                                    marker=dict(color='red', size=14, line=dict(width=3, color='black')))

        minVolRatio = go.Scatter(name='Minimum Volatility',
                                 mode='markers',
                                 x=[minVar_std],
                                 y=[minVar_ret],
                                 marker=dict(color='green', size=14, line=dict(width=3, color='black')))

        ef_curve = go.Scatter(name='Efficient Frontier',
                              mode='lines',
                              x=[round(ef_std * 100, 2) for ef_std in efficientFrontierList],
                              y=[round(target * 100, 2) for target in targetReturns],
                              line=dict(color='black', width=3, dash='dashdot'))

        data = [rfr, maxSharpeRatio, minVolRatio, ef_curve]
        layout = go.Layout(
            title='Portfolio Optimization with Efficient Frontier',
            yaxis=dict(title='Annualized Returns (%)'),
            xaxis=dict(title='Annualized Volatility (%)'),
            showlegend=True,
            legend=dict(x=0.70, y=0.0, traceorder='normal',
                        bgcolor='white',
                        bordercolor='black',
                        borderwidth=2),
            width=800, height=600)

        fig = go.Figure(data=data, layout=layout)
        return fig.show()

    def portFolioReturn(self, weights, meanReturns, covMatrix):
        return self.portfolioPerformance(weights, meanReturns, covMatrix)[0]

    def efficientOptimization(self, meanReturns, covMatrix, returnTarget, constrainSet=(0, 1)):
        """For each return target we want to optimize the portfolio to the min variance"""
        numAssets = len(meanReturns)
        args = (meanReturns, covMatrix)
        constraints = (
            {'type': 'eq', 'fun': lambda x: self.portFolioReturn(x, meanReturns, covMatrix) - returnTarget},
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bound = constrainSet
        bounds = (bound for asset in range(numAssets))
        effOpt = opt.minimize(self.portfolioVariance, numAssets * [1 / numAssets], args=args, method='SLSQP',
                              bounds=bounds, constraints=constraints)

        return effOpt
