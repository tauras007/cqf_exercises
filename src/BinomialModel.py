import numpy as np
import pandas as pd
from src.Option import Option


class BinomialModel:

    def __init__(self):
        initialOpt = Option(1, 100, 100, 'C', 0.2, 0.05, True)
        self.timestep = initialOpt.expiration / 4
        self.u = 1 + (initialOpt.volatility * np.sqrt(self.timestep))
        self.v = 1 - (initialOpt.volatility * np.sqrt(self.timestep))
        self.pdash = 0.5 + (initialOpt.interestRate * np.sqrt(self.timestep)) / (2 * initialOpt.volatility)
        self.discountFactor = 1 / (1 + initialOpt.interestRate * self.timestep)

        self.calculateCallOptionPayoff(initialOpt, 5)
        self.calculateOption(initialOpt)

    def calculateCallOptionPayoff(self, opt, depth):
        if depth <= 0:
            return

        self.replicate(opt)

        if opt.isUs:
            opt.payoff = np.maximum(((1 if opt.type == 'C' else -1) * (opt.stocks - opt.strike)), 0)
            if depth == 1:
                opt.option = opt.payoff

        depth_ = depth - 1
        self.calculateReplicate(opt.probable_paths, depth_)

    def calculateOption(self, opt: Option):
        while opt.option is None:
            if len(opt.probable_paths) > 0 and (opt.probable_paths[0].option or opt.probable_paths[1].option is not None):
                optVal = np.maximum(self.discountFactor * (
                        self.pdash * opt.probable_paths[0].option + (1 - self.pdash) * opt.probable_paths[
                    1].option), 0 * opt.payoff)
                delta  = (opt.probable_paths[0].option - opt.probable_paths[1].option)/(opt.probable_paths[0].stocks - opt.probable_paths[1].stocks)
                opt.option = optVal
                opt.delta = delta
                print('stocks', opt.stocks)
                print('payoff', opt.payoff)
                print('option', opt.option)
                print('delta', opt.delta)
                print('')
            else:
                for i in opt.probable_paths:
                    self.calculateOption(i)

    def calculateReplicate(self, optTuple: tuple, depth):
        for i in optTuple:
            self.calculateCallOptionPayoff(i, depth)

    def replicate(self, parent: Option):
        up = Option(parent.expiration, self.u * parent.stocks, parent.strike, parent.type, parent.volatility,
                    parent.interestRate, parent.isUs)
        down = Option(parent.expiration, self.v * parent.stocks, parent.strike, parent.type, parent.volatility,
                      parent.interestRate, parent.isUs)
        parent.probable_paths = (up, down)
