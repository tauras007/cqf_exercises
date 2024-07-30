import numpy as np
from numpy.linalg import multi_dot
import scipy as sp


class PortFolioOptExercise:

    def getData(self):
        """Sigma ie standard deviation of the assets in a diagonal 4x4 matrix """
        sigma = np.diag([0.12, 0.12, 0.15, 0.20])
        """Returns of the assets in a 1x4 matrix"""
        mu = np.array([0.08, 0.10, 0.10, 0.14])
        return mu, sigma

    def calculateWeights(self, mu, correlations, sigma, targetReturn):
        covariance = multi_dot([sigma, correlations, sigma])
        identityMatrix = np.array([1, 1, 1, 1])
        A = identityMatrix.transpose().dot(np.linalg.inv(covariance)).dot(identityMatrix)
        B = mu.transpose().dot(np.linalg.inv(covariance)).dot(identityMatrix)
        C = mu.transpose().dot(np.linalg.inv(covariance)).dot(mu)
        lamb_da = (A * targetReturn - B) / (A * C - np.square(B))
        gamma = (C - B * targetReturn) / (A * C - np.square(B))

        weights = np.linalg.inv(covariance).dot((mu.dot(lamb_da) + identityMatrix.dot(gamma)))
        return weights

    def portfolioReturns(self, mu, weights):
        return mu.transpose().dot(weights)

    def portfolioVolatility(self, sigma, correlations, weights):
        return np.sqrt(weights.transpose().dot(multi_dot([sigma, correlations, sigma])).dot(weights))

    def calcPortfolioReturnsAndVolatilityOnEfficientFrontier(self, mu, correlations, sigma, targetReturn):
        weights = self.onEfficientFrontier(mu, correlations, sigma, targetReturn)
        portfolioReturns = self.portfolioReturns(mu, weights)
        portfolioVolatility = self.portfolioVolatility(sigma, correlations, weights)
        returnLessThanTarget = sp.stats.norm.cdf(((portfolioReturns - targetReturn) / portfolioVolatility))

        return portfolioReturns, portfolioVolatility, returnLessThanTarget

    def onEfficientFrontier(self, mu, correlations, sigma, targetReturn):
        covariance = multi_dot([sigma, correlations, sigma])
        identityMatrix = np.array([1, 1, 1, 1])
        A = identityMatrix.transpose().dot(np.linalg.inv(covariance)).dot(identityMatrix)
        B = mu.transpose().dot(np.linalg.inv(covariance)).dot(identityMatrix)
        weights = np.linalg.inv(covariance).dot(mu - targetReturn * identityMatrix) / (B - A * targetReturn)
        return weights

    def isPositiveDefinite(self, correlations):
        eigvals = np.linalg.eigvals(correlations)
        print(eigvals)
        return np.all(eigvals > 0)
