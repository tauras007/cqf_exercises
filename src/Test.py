from src.LogNormalRandomWalk import LogNormalRandomWalk
from src.BinomialModel import BinomialModel
import numpy as np
from src.PortfolioOptimization import PortfolioOptimization
import datetime as dt

portfolio = PortfolioOptimization()
meanReturns, covMatrix, standardDev = portfolio.getData(['BARC.L', 'LLOY.L', 'NWG.L'], dt.datetime.now() - dt.timedelta(days=365*10),
                          dt.datetime.now())


results = portfolio.EE_Graph(meanReturns, covMatrix, 5.0)
print(results)
