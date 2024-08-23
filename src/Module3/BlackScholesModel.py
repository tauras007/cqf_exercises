import numpy as np
from pydantic import BaseModel, Field, computed_field
from scipy.stats import norm
from pydantic import BaseModel, Field, computed_field
from typing import Literal


class OptionInputs(BaseModel):
    option_type: Literal["call", "put"]
    spot_price: float = Field(gt=0, description="Current price of the underlying asset")
    strike_price: float = Field(gt=0, description="Strike price of the option")
    time_to_expiry: float = Field(gt=0, description="Time to expiration in years")
    risk_free_rate: float = Field(ge=0, le=1, description="Risk-free interest rate")
    volatility: float = Field(gt=0, description="Volatility of the underlying asset")


class BlackScholesModel(BaseModel):
    inputs: OptionInputs

    @computed_field
    def d1(self) -> float:
        return (np.log(self.inputs.spot_price / self.inputs.strike_price) +
                (self.inputs.risk_free_rate + 0.5 * self.inputs.volatility ** 2) * self.inputs.time_to_expiry) / \
            (self.inputs.volatility * np.sqrt(self.inputs.time_to_expiry))

    @computed_field
    def d2(self) -> float:
        return self.d1 - self.inputs.volatility * np.sqrt(self.inputs.time_to_expiry)

    @computed_field
    def price(self) -> float:
        if self.inputs.option_type == "call":
            return self.inputs.spot_price * norm.cdf(self.d1) - \
                self.inputs.strike_price * np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(
                    self.d2)
        else:  # put option
            return self.inputs.strike_price * np.exp(
                -self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(-self.d2) - \
                self.inputs.spot_price * norm.cdf(-self.d1)

    @computed_field
    def delta(self) -> float:
        if self.inputs.option_type == "call":
            return norm.cdf(self.d1)
        else:  # put option
            return norm.cdf(self.d1) - 1

    @computed_field
    def gamma(self) -> float:
        return norm.pdf(self.d1) / (
                self.inputs.spot_price * self.inputs.volatility * np.sqrt(self.inputs.time_to_expiry))

    @computed_field
    def vega(self) -> float:
        return self.inputs.spot_price * norm.pdf(self.d1) * np.sqrt(self.inputs.time_to_expiry) / 100

    @computed_field
    def theta(self) -> float:
        common = -(self.inputs.spot_price * norm.pdf(self.d1) * self.inputs.volatility) / (
                2 * np.sqrt(self.inputs.time_to_expiry))
        if self.inputs.option_type == "call":
            return (common - self.inputs.risk_free_rate * self.inputs.strike_price *
                    np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(self.d2)) / 365
        else:  # put option
            return (common + self.inputs.risk_free_rate * self.inputs.strike_price *
                    np.exp(-self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(-self.d2)) / 365

    @computed_field
    def rho(self) -> float:
        if self.inputs.option_type == "call":
            return self.inputs.strike_price * self.inputs.time_to_expiry * np.exp(
                -self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(self.d2) / 100
        else:  # put option
            return -self.inputs.strike_price * self.inputs.time_to_expiry * np.exp(
                -self.inputs.risk_free_rate * self.inputs.time_to_expiry) * norm.cdf(-self.d2)


def getcalloption(option_type, spot_price, strike_price, time_to_expiry, risk_free_rate, volatility):
    return BlackScholesModel(inputs=OptionInputs(option_type=option_type,spot_price=spot_price,strike_price=strike_price,
                                                 time_to_expiry=time_to_expiry,risk_free_rate=risk_free_rate,volatility=volatility))
