from pydantic import BaseModel, Field, PositiveFloat, PositiveInt
import numpy as np
from scipy.stats import qmc
import scipy.stats as stats
from typing import Literal


# Step 1: Define the Input Model using Pydantic
class AsianOptionInput(BaseModel):
    spot_price: PositiveFloat = Field(..., description="Current price of the underlying asset (S0)")
    strike_price: PositiveFloat = Field(..., description="Strike price of the option (K)")
    risk_free_rate: float = Field(..., description="Risk-free interest rate (r)")
    volatility: PositiveFloat = Field(..., description="Volatility of the underlying asset (σ)")
    maturity: PositiveFloat = Field(..., description="Time to maturity in years (T)")
    num_time_steps: PositiveInt = Field(..., description="Number of time steps for simulation")
    num_paths: PositiveInt = Field(..., description="Number of Monte Carlo paths")
    option_type: Literal['call', 'put'] = Field(..., description="Type of Asian option: 'call' or 'put'")


# Step 2: Sobol Sequence Generator and Monte Carlo Simulation
def generate_sobol_paths(input_params: AsianOptionInput) -> np.ndarray:
    """Generate Monte Carlo paths using Sobol sequences."""
    # Time increment
    dt = input_params.maturity / input_params.num_time_steps

    # Sobol sequence generator
    sobol = qmc.Sobol(d=input_params.num_time_steps, scramble=True)

    # Generate Sobol sequence samples
    sobol_samples = sobol.random(input_params.num_paths)

    # Transform Sobol samples to standard normal random variables
    normal_samples = qmc.scale(sobol_samples, 0, 1)
    normal_samples = stats.norm.ppf(normal_samples)  # Convert to normal distribution

    # Initialize asset price paths
    paths = np.zeros((input_params.num_paths, input_params.num_time_steps + 1))
    paths[:, 0] = input_params.spot_price

    # Simulate paths
    for t in range(1, input_params.num_time_steps + 1):
        dW = normal_samples[:, t - 1] * np.sqrt(dt)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (input_params.risk_free_rate - 0.5 * input_params.volatility ** 2) * dt + input_params.volatility * dW)

    return paths


# Step 3: Asian Option Pricing Function
def price_asian_option(input_params: AsianOptionInput) -> float:
    """Price an Asian option using Monte Carlo simulation with Sobol sequences."""
    paths = generate_sobol_paths(input_params)

    # Calculate the arithmetic average of the paths
    arithmetic_averages = np.mean(paths[:, 1:], axis=1)  # Exclude the initial spot price

    # Calculate the payoff
    if input_params.option_type == 'call':
        payoffs = np.maximum(arithmetic_averages - input_params.strike_price, 0)
    else:  # Put option
        payoffs = np.maximum(input_params.strike_price - arithmetic_averages, 0)

    # Discounted payoff
    discounted_payoff = np.exp(-input_params.risk_free_rate * input_params.maturity) * np.mean(payoffs)

    return discounted_payoff


# Step 4: Run the Monte Carlo Simulation with Example Input
if __name__ == "__main__":
    # Define input parameters
    input_params = AsianOptionInput(
        spot_price=100,
        strike_price=100,
        risk_free_rate=0.05,
        volatility=0.2,
        maturity=1,
        num_time_steps=100,
        num_paths=10000,
        option_type='call'
    )

    # Price the Asian option
    option_price = price_asian_option(input_params)
    print(f"The price of the Asian {input_params.option_type} option is: {option_price:.4f}")
