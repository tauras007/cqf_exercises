
class Option:

    def __init__(self, expiration: int, stocks: int, strike: float, optionType: str, volatility: float, interest_rate: float, isUs: bool):
        self.expiration = expiration
        self.stocks = stocks
        self.strike = strike
        self.type = optionType
        self.volatility = volatility
        self.interestRate = interest_rate
        self.isUs = isUs
        self.payoff = None
        self.delta = None
        self.option = None
        self.probable_paths = ()

