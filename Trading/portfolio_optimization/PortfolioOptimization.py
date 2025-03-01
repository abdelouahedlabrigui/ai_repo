from matplotlib import pyplot as plt
import numpy as np
import yfinance as yf
import pandas as pd
from scipy.optimize import minimize
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import itertools

class PortfolioOptimization:
    def __init__(self, tickers: list, start_date: str, end_date: str):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = yf.download(tickers, start=self.start_date, end=self.end_date)["Close"]
        self.param_grid = {
            "learning_rate": [1e-4, 3e-4, 1e-3],  
            "batch_size": [64, 128, 256],  
            "gamma": [0.95, 0.99, 0.999],  
            "clip_range": [0.1, 0.2, 0.3],  
            "n_steps": [512, 1024, 2048],  
            "gae_lambda": [0.9, 0.95, 0.99]
        }
        self.param_combinations = list(itertools.product(*self.param_grid.values()))
        self.best_params = None
        self.best_reward = -np.inf
        self.ppo_params = {
            "learning_rate": 3e-4,
            "batch_size": 128,
            "gamma": 0.99,
            "clip_range": 0.2,
            "n_steps": 1024,
            "gae_lambda": 0.95,
            "verbose": 1
        }


    def returns_and_risk_metrics(self):
        df = self.data
        returns = np.log(df / df.shift(1)).dropna()
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        return returns, mean_returns, cov_matrix
    
    def portfolio_performance(self, weights, mean_returns, cov_matrix, risk_free_rate=0.02):
        returns = np.dot(weights, mean_returns)
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (returns - risk_free_rate) / std
        return -sharpe_ratio

    def optimize_portfolio(self):
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()
        constraints = {"type": "eq", "fun": lambda x: np.sum(x) - 1}
        bounds = tuple((0, 1) for _ in range(len(self.tickers)))
        init_weights = np.array([1/len(self.tickers)] * len(self.tickers))
        optimized = minimize(self.portfolio_performance, init_weights, args=(mean_returns, cov_matrix),
                             method="SLSQP", bounds=bounds, constraints=constraints)
        optimal_weights = optimized.x
        return optimal_weights, optimized
    
    def simulate_rl_based_strategy(self):
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()
        optimal_weights, optimized = self.optimize_portfolio()
        env = DummyVecEnv([lambda: PortfolioEnv(returns, self.tickers)])
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=50000)
        model.save("portfolio_rl_model")
        markowitz_returns = (returns @ optimal_weights).cumsum()
        r1_allocations = []
        state = env.reset()
        for i in range(len(returns)):
            action, _ = model.predict(state)
            r1_allocations.append(action)
            state, _, _, _ = env.step(action)
        r1_weights = np.array(r1_allocations).squeeze()
        rl_returns = (returns.values * r1_weights).sum(axis=1).cumsum()
        return markowitz_returns, rl_returns

    def portfolio_performance_vs_markowitz(self):
        markowitz_returns, rl_returns = self.simulate_rl_based_strategy()
        plt.figure(figsize=(12, 6))
        plt.plot(markowitz_returns, label="Markowitz Strategy", color="blue")
        plt.plot(rl_returns, label="RL Strategy", color="red")
        plt.title("Portfolio Performance: RL vs Markowitz")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def hyperparameter_combinations(self):
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()
        env = DummyVecEnv([lambda: PortfolioEnv(returns, self.tickers)])
        for params in self.param_combinations:
            lr, batch_size, gamma, clip_range, n_steps, gae_lambda = params
            model = PPO("MlpPolicy", env, learning_rate=lr, batch_size=batch_size, gamma=gamma, clip_range=clip_range, n_steps=n_steps,
                        gae_lambda=gae_lambda, verbose=0)
            model.learn(total_timesteps=50000)

            obs = env.reset()
            total_reward = 0
            for i in range(len(returns)):
                action, _ = model.predict(obs)
                obs, reward, done, _ = env.step(action)
                total_reward += reward[0]
            
            if total_reward > self.best_reward:
                best_reward = total_reward
                best_params = params

        return env, best_params, best_reward

    def simulate_optimized_ppo_strategy(self):        
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()
        env, best_params, best_reward = self.hyperparameter_combinations()
        best_hyperparams = dict(zip(self.param_grid.keys(), best_params))
        optimal_model = PPO("MlpPolicy", env,
                            learning_rate=best_hyperparams["learning_rate"],
                            batch_size=best_hyperparams["batch_size"],
                            gamma=best_hyperparams["gamma"],
                            clip_range=best_hyperparams["clip_range"],
                            n_steps=best_hyperparams["n_steps"],
                            gae_lambda=best_hyperparams["gae_lambda"], verbose=1)
        optimal_model.learn(total_timesteps=100000)
        optimal_model.save("optimal_portfolio_r1")
        obs = env.reset()
        r1_opt_allocations = []
        for i in range(len(returns)):
            action, _ = optimal_model.predict(obs)
            r1_opt_allocations.append(obs)
            obs, _, _, _ = env.step(action)
        
        r1_opt_weights = np.array(r1_opt_allocations).squeeze()
        rl_opt_returns = (returns.values * r1_opt_weights).sum(axis=1).cumsum()
        return rl_opt_returns
    
    def rl_portfolio_performance_tuned_vs_default_ppo(self):
        markowitz_returns, rl_returns = self.simulate_rl_based_strategy()
        rl_opt_returns = self.simulate_optimized_ppo_strategy()

        plt.figure(figsize=(12, 6))
        plt.plot(rl_opt_returns, label="Optimized RL Strategy", color="red", linewidth=2)
        plt.plot(rl_returns, label="Default RL Strategy", color="blue", linestyle="dashed")
        plt.title("RL Portfolio Performance: Tuned vs Defaul PPO")
        plt.xlabel("Cumulative Return")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def simulate_optimized_ppo_strategy(self):
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()        
        env = DummyVecEnv([lambda: PortfolioEnvV1(returns, self.tickers, max_drawdown_limit=0.2, volatility_limit=0.3)])
        model = PPO("MlpPolicy", env, **self.ppo_params)
        model.learn(total_timesteps=100000)
        model.save("risk_aware_portfolio_rl")
        obs = env.reset()
        rl_opt_allocations = []
        for i in range(len(returns)):
            action, _ = model.predict(obs)
            rl_opt_allocations.append(action)
            obs, _, _, _ = env.step(action)

        rl_opt_weights = np.array(rl_opt_allocations).squeeze()
        rl_opt_returns = (returns.values * rl_opt_weights).sum(axis=1).cumsum()
        return rl_opt_returns
    
    def plot_risk_aware_rl_portfolio_optimization(self):
        rl_opt_returns = self.simulate_optimized_ppo_strategy()
        markowitz_returns, rl_returns = self.simulate_rl_based_strategy()

        plt.figure(figsize=(12, 6))
        plt.plot(rl_opt_returns, label="Risk wre RL Strategy", color="green", linewidth=2)
        plt.plot(rl_returns, label="Defaul RL Strategy", color="red", linewidth="dashed")
        plt.title("Risk Awre RL Portfolio Optimization")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def simulate_rl_strategy_risk_free_asset(self):
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()    
        env = DummyVecEnv([lambda: PortfolioEnvWithRiskFree(returns, self.tickers, risk_free_rate=0.03)])
        model = PPO("MlpPolicy", env, **self.ppo_params)
        model.learn(total_timesteps=100000)
        model.save("risk_aware_portfolio_rl_with_bonds")
        obs = env.reset()
        rl_opt_allocations = []
        for i in range(len(returns)):
            action, _ = model.predict(obs)
            rl_opt_allocations.append(action)
            obs, _, _, _ = env.step(action)
        
        rl_opt_weights = np.array(rl_opt_allocations).squeeze()
        rl_opt_returns = (returns.values * rl_opt_weights[:,:-1]).sum(axis=1).cumsum()
        return rl_opt_returns
    
    def plot_portfolio_optimiation_risk_free_asset(self):
        markowitz_returns, rl_returns = self.simulate_rl_based_strategy()
        rl_opt_returns = self.simulate_rl_strategy_risk_free_asset()
        plt.figure(figsize=(12, 6))
        plt.plot(rl_opt_returns, label="Risk-Free RL Strategy", color="blue", linewdith=2)
        plt.plot(rl_returns, label="Risk-Only RL Strategy", color="red", linestyle="dashed")
        plt.title("Portfolio Optimization with Risk-Free Asset")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def simulate_diversified_rl_strategy(self):
        returns, mean_returns, cov_matrix = self.returns_and_risk_metrics()    
        env = DummyVecEnv([lambda: PortfolioEnvWithContraints(returns, self.tickers, risk_free_rate=0.03)])
        model = PPO("MlpPolicy", env, **self.ppo_params)
        model.learn(total_timesteps=100000)
        model.save("diversified_portfolio_rl")
        obs = env.reset()
        rl_div_allocations = []
        for i in range(len(returns)):
            action, _ = model.predict(obs)
            rl_div_allocations.append(action)
            obs, _, _, _ = env.step(action)

        rl_div_weights = np.array(rl_div_allocations).squeeze()
        rl_div_returns = (returns.values * rl_div_weights[:, :-1]).sum(axis=1).cumsum()

        return rl_div_returns
    
    def plot_portfolio_optimization_diversification_constraints(self):
        rl_opt_returns = self.simulate_optimized_ppo_strategy()
        rl_div_returns = self.simulate_diversified_rl_strategy()

        plt.figure(figsize=(12, 6))
        plt.plot(rl_div_returns, label="Diversified RL Stratgey", color="green", linewidth=2)
        plt.plot(rl_opt_returns, label="Unconstrained RL Strategy", color="red", linstyle="dashed")
        plt.title("Portfolio Optimization with Diversification Constraints")
        plt.xlabel("Days")
        plt.ylabel("Cumulative Returns")
        plt.legend()
        plt.show()



class PortfolioEnv(gym.Env):
    def __init__(self, df, tickers, initial_balance=10000):
        super(PortfolioEnv, self).__init__()
        self.df = df
        self.tickers = tickers
        self.initial_balance = initial_balance
        self.current_step = 0
        self.weights = np.array([1/len(tickers)] * len(tickers))

        # Action space: Portfolio weights (must sum to 1)
        self.action_space = spaces.Box(low=0, high=1, shape=(len(tickers),), dtype=np.float32)
        
        # Observation space: Stock returns
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(tickers),), dtype=np.float32)

    def step(self, action):
        # Ensure action values are positive
        action = np.clip(action, 1e-6, 1)  # Avoid zero allocations

        # Normalize action weights to sum to 1
        action /= np.sum(action) + 1e-6  # Add small epsilon

        # Store the new weights
        self.weights = action  

        # Calculate portfolio return
        returns = self.df.iloc[self.current_step].values
        portfolio_return = np.dot(self.weights, returns)

        # Move to the next step
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Reward function (Sharpe Ratio Approximation)
        rolling_std = np.std(self.df.iloc[max(0, self.current_step-30):self.current_step].values) + 1e-6
        reward = portfolio_return / rolling_std  # Adjusted reward

        return self.weights, reward, done, {}

    def reset(self):
        self.current_step = 0
        return np.array(self.df.iloc[self.current_step].values)

    
    

class PortfolioEnvV1(gym.Env):
    def __init__(self, returns, tickers, max_drawdown_limit=0.2, volatility_limit=0.3):
        super(PortfolioEnvV1, self).__init__()
        self.returns = returns
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.current_step =0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_limit = volatility_limit
        self.history = []

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.obeservation_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= np.sum(action)

        portfolio_return = np.dot(action, self.returns.iloc[self.current_step])
        self.portfolio_value *= (1 + portfolio_return)
        self.history.append(self.portfolio_value)

        peak = max(self.history)
        drawdown = (peak - self.portfolio_value) / peak if peak > 0 else 0
        max_drawdown_penality = max(0, drawdown - self.max_drawdown_limit)

        if len(self.history) > 1:
            volatility = np.std(self.history[-30:])
        else:
            volatility = 0
        
        volatility_penality = max(0, volatility - self.volatility_limit)

        sharpe_ratio = portfolio_return / (volatility + 1e-6)
        reward = sharpe_ratio - 0.1 * max_drawdown_penality - 0.1 * volatility_penality

        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return self.returns.iloc[self.current_step].values, reward, done, {}
    
    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        self.history = [1.0]
        return self.returns.iloc[self.current_step].values
    
class PortfolioEnvWithRiskFree(gym.Env):
    def __init__(self, returns, tickers, risk_free_rate=0.03, max_drawdown_limit=0.2, volatility_limit=0.3):
        super(PortfolioEnvWithRiskFree, self).__init__()
        self.returns = returns
        self.tickers = tickers
        self.n_assets = len(tickers) + 1
        self.current_step = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.risk_free_rate = 1.0
        self.max_drawdown_limit = max_drawdown_limit
        self.volatility_limit = volatility_limit
        self.history = []

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= np.sum(action)
        risky_return = np.dot(action[:-1], self.returns.iloc[self.current_step])
        risk_free_weight = action[-1]
        total_return = (1 - risk_free_weight) * risky_return + risk_free_weight * (self.risk_free_rate / 252)
        self.portfolio_value *= (1 + total_return)
        self.history.append(self.portfolio_value)
        peak = max(self.history)
        drawdown = (peak - self.portfolio_value) / peak if peak > 0 else 0
        max_drawdown_penality = max(0, drawdown - self.max_drawdown_limit)

        if len(self.history) > 1:
            volatility = np.std(self.history[-30:])
        else:
            volatility = 0

        volatility_penality = max(0, volatility - self.volatility_limit)

        sharpe_ratio = total_return / (volatility + 1e-6)

        reward = sharpe_ratio - 0.1 * max_drawdown_penality - 0.1 * volatility_penality

        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return np.append(self.returns.iloc[self.current_step].vlues, self.risk_free_rate / 252), reward, done, {}
    
    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        self.history = [1.0]
        return np.append(self.returns.iloc[self.current_step].values, self.risk_free_rate / 252)
         

class PortfolioEnvWithContraints(gym.Env):
    def __init__(self, returns, tickers, risk_free_rate=0.03, max_weight=0.4, min_sector_alloc=0.2):
        super(PortfolioEnvWithContraints, self).__init__()

        self.returns = returns
        self.tickers = tickers
        self.n_assets = len(tickers)
        self.current_step = 0
        self.weights = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_sector_allow = min_sector_alloc

        self.sector_map = {
            "Technology": [0, 1], "Finance": [2], "Energy":[3]
        }

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_assets,), dtype=np.float32)

    def step(self, action):
        action = np.clip(action, 0, 1)
        action /= np.sum(action)
        action[:-1] = np.minimum(action[:-1], self.max_weight)
        for sector, indices in self.sector_map.items():
            sector_alloc = np.sum(action[indices])
            if sector_alloc < self.min_sector_allow:
                action[indices] += (self.min_sector_allow - sector_alloc) / len(indices)

        risky_return = np.dot(action[:-1], self.returns.iloc[self.current_step])
        risk_free_weight = action[-1]
        total_return = (1 - risk_free_weight) * risky_return + risk_free_weight * (self.risk_free_rate / 252)

        self.portfolio_value *= (1 + total_return)
        sharpe_ratio = total_return / (np.std(self.returns.iloc[max(0, self.current_step-30):self.current_step]) + 1e-6)
        reward = sharpe_ratio
        self.current_step += 1
        done = self.current_step >= len(self.returns) - 1
        return np.append(self.returns.iloc[self.current_step].values, self.risk_free_rate / 252), reward, done, {}
    
    def reset(self):
        self.current_step = 0
        self.portfolio_value = 1.0
        return np.append(self.returns.iloc[self.current_step].values, self.risk_free_rate / 252)
    
if __name__ == "__main__":
    tickers: list = ['AAPL', 'MSFT', 'JPM', 'XOM']
    start_date: str = '2023-01-01'
    end_date: str = '2023-12-31'
    portfolio = PortfolioOptimization(tickers, start_date, end_date)
    returns, mean_returns, cov_matrix = portfolio.returns_and_risk_metrics()
    print("Expected Annual Returns:\n", mean_returns)
    print("\nCovariance Matrix:\n", cov_matrix)
    optimal_weights, optimized = portfolio.optimize_portfolio()
    print("\nOptimal Portfolio Weights:\n", dict(zip(tickers, optimal_weights)))
    portfolio.portfolio_performance_vs_markowitz()
    env, best_params, best_reward = portfolio.hyperparameter_combinations()
    print(f"Best Hyperparameters: {dict(zip(portfolio.param_grid.keys(), best_params))}")
    print(f"Best Cumulative Reward: {best_reward}")
    portfolio.rl_portfolio_performance_tuned_vs_default_ppo()
    portfolio.plot_risk_aware_rl_portfolio_optimization()
    portfolio.plot_portfolio_optimiation_risk_free_asset()
    portfolio.plot_portfolio_optimization_diversification_constraints()