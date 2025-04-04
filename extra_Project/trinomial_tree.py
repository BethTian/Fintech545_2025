import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.stats import norm
import os

class TrinomialTreeModel:
    def __init__(self, S0, K, r, sigma, T, is_call=True, q=0.0):
        """
        Initialize the trinomial tree model parameters
        
        Parameters:
        S0 (float): Initial stock price
        K (float): Strike price
        r (float): Risk-free interest rate (annualized)
        sigma (float): Volatility (annualized)
        T (float): Time to maturity (in years)
        is_call (bool): Option type (True for call, False for put)
        q (float): Dividend yield (annualized)
        """
        self.S0 = S0
        self.K = K
        self.r = r
        self.sigma = sigma
        self.T = T
        self.is_call = is_call
        self.q = q
        
    def _setup_parameters(self, n):
        """
        Set up the trinomial tree parameters
        
        Parameters:
        n (int): Number of time steps
        
        Returns:
        tuple: (dt, u, d, pu, pm, pd) - time step, up factor, down factor, up probability, middle probability, down probability
        """
        dt = self.T / n
        # The "stretch" parameter - typically sqrt(3) for trinomial trees
        stretch = np.sqrt(3)
        
        # Calculate up and down factors
        u = np.exp(self.sigma * np.sqrt(dt) * stretch)
        d = 1 / u
        
        # Standard trinomial tree parameterization
        a = np.exp((self.r - self.q) * dt)
        pu = ((a - d) / (u - d) + (u*a - 1) / (u - 1) - 1) / 2  # Probability of up move
        pd = ((u - a) / (u - d) + (1 - a) / (u - 1) - 1) / 2    # Probability of down move
        pm = 1 - pu - pd                                        # Probability of middle move
        
        # Check for valid probabilities
        if not (0 <= pu <= 1 and 0 <= pm <= 1 and 0 <= pd <= 1):
            print(f"Warning: Invalid probabilities calculated: pu={pu}, pm={pm}, pd={pd}")
            print("Using alternative parameterization")
            
            # Alternative simpler parameterization if the above fails
            pu = (1/6) * (1 + np.sqrt(dt) * ((self.r - self.q)/self.sigma - 0.5 * self.sigma))
            pd = (1/6) * (1 - np.sqrt(dt) * ((self.r - self.q)/self.sigma - 0.5 * self.sigma))
            pm = 2/3
        
        return dt, u, d, pu, pm, pd
    
    def _build_stock_price_tree(self, n, u, d):
        """
        Build the stock price tree
        
        Parameters:
        n (int): Number of time steps
        u (float): Up factor
        d (float): Down factor
        
        Returns:
        numpy.ndarray: Stock price tree
        """
        # The tree has 2n+1 nodes at the final level
        stock_tree = np.zeros((2*n+1, n+1))
        # Set initial stock price
        stock_tree[n, 0] = self.S0
        
        # Build the tree
        for j in range(1, n+1):
            for i in range(2*n+1):
                # Skip nodes not reachable
                if stock_tree[i, j-1] == 0 and i != n:
                    continue
                    
                # Up move
                if i > 0:  # Ensure we don't go out of bounds
                    stock_tree[i-1, j] = stock_tree[i, j-1] * u
                
                # Middle move
                stock_tree[i, j] = stock_tree[i, j-1]
                
                # Down move
                if i < 2*n:  # Ensure we don't go out of bounds
                    stock_tree[i+1, j] = stock_tree[i, j-1] * d
        
        return stock_tree
    
    def _calculate_option_price_tree(self, stock_tree, dt, pu, pm, pd, n, american=False):
        """
        Calculate the option price tree
        
        Parameters:
        stock_tree (numpy.ndarray): Stock price tree
        dt (float): Time step
        pu (float): Probability of up move
        pm (float): Probability of middle move
        pd (float): Probability of down move
        n (int): Number of time steps
        american (bool): True for American option, False for European option
        
        Returns:
        numpy.ndarray: Option price tree
        """
        # Initialize option price tree
        option_tree = np.zeros_like(stock_tree)
        
        # Calculate option payoff at maturity
        for i in range(2*n+1):
            if stock_tree[i, n] > 0:  # Only calculate for nodes with positive stock price
                if self.is_call:
                    option_tree[i, n] = max(0, stock_tree[i, n] - self.K)
                else:
                    option_tree[i, n] = max(0, self.K - stock_tree[i, n])
        
        # Backward induction
        discount_factor = np.exp(-self.r * dt)
        
        for j in range(n-1, -1, -1):
            for i in range(2*n+1):
                # Skip nodes that are not part of the tree
                if stock_tree[i, j] == 0:
                    continue
                
                # Ensure indices are valid for the neighbors
                valid_up = 0 <= i-1 < 2*n+1
                valid_middle = 0 <= i < 2*n+1
                valid_down = 0 <= i+1 < 2*n+1
                
                # Calculate expected option value
                expected_value = 0
                if valid_up and stock_tree[i-1, j+1] > 0:
                    expected_value += pu * option_tree[i-1, j+1]
                if valid_middle and stock_tree[i, j+1] > 0:
                    expected_value += pm * option_tree[i, j+1]
                if valid_down and stock_tree[i+1, j+1] > 0:
                    expected_value += pd * option_tree[i+1, j+1]
                
                expected_value *= discount_factor
                
                if american:
                    # For American options, consider early exercise
                    if self.is_call:
                        intrinsic_value = max(0, stock_tree[i, j] - self.K)
                    else:
                        intrinsic_value = max(0, self.K - stock_tree[i, j])
                    option_tree[i, j] = max(expected_value, intrinsic_value)
                else:
                    # For European options, just use the expected value
                    option_tree[i, j] = expected_value
        
        return option_tree
    
    def price_option(self, n=100, american=False):
        """
        Price the option using a trinomial tree model
        
        Parameters:
        n (int): Number of time steps
        american (bool): True for American option, False for European option
        
        Returns:
        float: Option price
        """
        start_time = time.time()
        
        # Set up parameters
        dt, u, d, pu, pm, pd = self._setup_parameters(n)
        
        # Build stock price tree
        stock_tree = self._build_stock_price_tree(n, u, d)
        
        # Calculate option price tree
        option_tree = self._calculate_option_price_tree(stock_tree, dt, pu, pm, pd, n, american)
        
        # Option price is at the root of the tree
        option_price = option_tree[n, 0]
        
        end_time = time.time()
        computation_time = end_time - start_time
        
        return option_price, computation_time
    
    def black_scholes_price(self):
        """
        Calculate the option price using the Black-Scholes formula (for comparison)
        
        Returns:
        float: Option price according to Black-Scholes
        """
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if self.is_call:
            return self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
    
    def analyze_convergence(self, max_steps=200, step_size=10):
        """
        Analyze the convergence of the trinomial tree model to the Black-Scholes price
        
        Parameters:
        max_steps (int): Maximum number of time steps to consider
        step_size (int): Step size for the number of time steps
        
        Returns:
        tuple: (steps, european_prices, american_prices, european_times, american_times, bs_price)
        """
        steps = list(range(step_size, max_steps + 1, step_size))
        european_prices = []
        american_prices = []
        european_times = []
        american_times = []
        
        bs_price = self.black_scholes_price()
        
        for n in steps:
            # European option
            eu_price, eu_time = self.price_option(n=n, american=False)
            european_prices.append(eu_price)
            european_times.append(eu_time)
            
            # American option
            am_price, am_time = self.price_option(n=n, american=True)
            american_prices.append(am_price)
            american_times.append(am_time)
        
        return steps, european_prices, american_prices, european_times, american_times, bs_price
    
    def plot_convergence(self, max_steps=200, step_size=10, save_path=None):
        """
        Plot the convergence of the trinomial tree model to the Black-Scholes price
        
        Parameters:
        max_steps (int): Maximum number of time steps to consider
        step_size (int): Step size for the number of time steps
        save_path (str): Directory path to save the plots, if None plots are displayed but not saved
        """
        steps, european_prices, american_prices, european_times, american_times, bs_price = self.analyze_convergence(max_steps, step_size)
        
        # Price convergence plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(steps, european_prices, 'b-', label='European Option')
        plt.plot(steps, american_prices, 'r-', label='American Option')
        plt.axhline(y=bs_price, color='g', linestyle='--', label='Black-Scholes')
        plt.xlabel('Number of Steps')
        plt.ylabel('Option Price')
        plt.title('Price Convergence')
        plt.legend()
        plt.grid(True)
        
        # Price error plot
        plt.subplot(1, 2, 2)
        plt.plot(steps, [abs(price - bs_price) for price in european_prices], 'b-', label='European Option Error')
        plt.xlabel('Number of Steps')
        plt.ylabel('Absolute Error (vs. Black-Scholes)')
        plt.title('Convergence Error')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.tight_layout()
        
        # Save price convergence plot if path is provided
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'price_convergence.png'), dpi=300, bbox_inches='tight')
            print(f"Price convergence plot saved to {os.path.join(save_path, 'price_convergence.png')}")
        
        plt.show()
        
        # Computational complexity plot
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(steps, european_times, 'b-', label='European Option')
        plt.plot(steps, american_times, 'r-', label='American Option')
        plt.xlabel('Number of Steps')
        plt.ylabel('Computation Time (s)')
        plt.title('Computational Time')
        plt.legend()
        plt.grid(True)
        
        # Complexity analysis
        plt.subplot(1, 2, 2)
        plt.loglog(steps, european_times, 'b-', label='European Option')
        plt.loglog(steps, american_times, 'r-', label='American Option')
        plt.loglog(steps, [10**-6 * n**2 for n in steps], 'g--', label='O(n²) Reference')
        plt.xlabel('Number of Steps (log scale)')
        plt.ylabel('Computation Time (s) (log scale)')
        plt.title('Computational Complexity')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        # Save computational complexity plot if path is provided
        if save_path is not None:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, 'computational_complexity.png'), dpi=300, bbox_inches='tight')
            print(f"Computational complexity plot saved to {os.path.join(save_path, 'computational_complexity.png')}")
        
        plt.show()

# Example usage
if __name__ == "__main__":
    # Option parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    r = 0.05    # Risk-free rate
    sigma = 0.2  # Volatility
    T = 1.0     # Time to maturity (in years)
    q = 0.02    # Dividend yield
    
    # Create model instances for call and put options
    call_model = TrinomialTreeModel(S0, K, r, sigma, T, is_call=True, q=q)
    put_model = TrinomialTreeModel(S0, K, r, sigma, T, is_call=False, q=q)
    
    # Number of steps
    n = 100
    
    # Price European options
    eu_call_price, _ = call_model.price_option(n=n, american=False)
    eu_put_price, _ = put_model.price_option(n=n, american=False)
    
    # Price American options
    am_call_price, _ = call_model.price_option(n=n, american=True)
    am_put_price, _ = put_model.price_option(n=n, american=True)
    
    # Black-Scholes prices (for comparison)
    bs_call_price = call_model.black_scholes_price()
    bs_put_price = put_model.black_scholes_price()
    
    # Print results
    print(f"European Call Option Price (Trinomial Tree, n={n}): {eu_call_price:.4f}")
    print(f"European Call Option Price (Black-Scholes): {bs_call_price:.4f}")
    print(f"European Call Option Price Difference: {abs(eu_call_price - bs_call_price):.6f}")
    print()
    
    print(f"European Put Option Price (Trinomial Tree, n={n}): {eu_put_price:.4f}")
    print(f"European Put Option Price (Black-Scholes): {bs_put_price:.4f}")
    print(f"European Put Option Price Difference: {abs(eu_put_price - bs_put_price):.6f}")
    print()
    
    print(f"American Call Option Price (Trinomial Tree, n={n}): {am_call_price:.4f}")
    print(f"American Put Option Price (Trinomial Tree, n={n}): {am_put_price:.4f}")
    print()
    
    # Early exercise premium
    call_premium = am_call_price - eu_call_price
    put_premium = am_put_price - eu_put_price
    print(f"Call Option Early Exercise Premium: {call_premium:.4f}")
    print(f"Put Option Early Exercise Premium: {put_premium:.4f}")
    
    # Directory to save plots
    save_dir = "/home/bethtian/fintech545/beth-fintech545/extra_Project"  # Current directory, change this to your preferred directory
    
    # Analyze convergence and computational complexity, and save plots
    print("\nAnalyzing convergence and computational complexity...")
    call_model.plot_convergence(max_steps=150, step_size=10, save_path=save_dir)