from src.LogNormalRandomWalk import LogNormalRandomWalk
from src.BinomialModel import BinomialModel
import numpy as np
from src.PortfolioOptimization import PortfolioOptimization
import datetime as dt
from src.Module2.Module2Exercise import PortFolioOptExercise as PortFolioOptExercise

#
# portfolio = PortfolioOptimization()
# meanReturns, covMatrix, standardDev = portfolio.getData(['BARC.L', 'LLOY.L', 'NWG.L'], dt.datetime.now() - dt.timedelta(days=365*10),
#                           dt.datetime.now())
#
#
# results = portfolio.EE_Graph(meanReturns, covMatrix, 5.0)
# print(results)

popt = PortFolioOptExercise()
returns, stdMatrix = popt.getData()
cov1 = np.array([[1, 0.2, 0.5, 0.3],
                 [0.2, 1, 0.7, 0.4],
                 [0.5, 0.7, 1, 0.9],
                 [0.3, 0.4, 0.9, 1]])
cov2 = np.diag([1, 1, 1, 1])
"""Since the square matrix whose determinant is zero is a singular matrix, it cannot be 
inverted and thus we cannot have a solution"""
cov3 = np.ones((4, 4))
print(popt.calcPortfolioReturnsAndVolatilityOnEfficientFrontier(returns, cov1, stdMatrix, 0.05))
print(popt.onEfficientFrontier(returns, cov1, stdMatrix, 0.05))


# corr = np.matrix([[9, 3, 0],
#                 [3, 16, 5],
#                 [0, 5, 25]])
# popt.isPositiveDefinite(corr)
