import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import norm
from scipy.special import gammaln


import wrds
from user_name import WRDS_USERNAME
db = wrds.Connection(wrds_username=WRDS_USERNAME)

# -------------------------------------------------------------------------------------------

def crsp_data(tickers_array, start_year, end_year):
    """
    Returns a dictionary per ticker with CRSP data.
    
    Given:
    - tickers_array: array of tickers
    - start_year   : starting year of time range
    - end_year     : ending year of time range
    """
    # Convert to SQL input
    tickers = np.sort(tickers_array)
    tickers = ', '. join(f"'{ticker}'" for ticker in tickers)
    
    # Get CRSP dataframe
    df = db.raw_sql(f"""
    SELECT dsf.hsiccd, dsf.permno, sn.comnam, sn.ticker, sn.shrcd,
           dsf.date, dsf.prc, dsf.vol, dsf.ret
    FROM crsp_a_stock.dsf AS dsf
    LEFT JOIN crsp_a_stock.stocknames AS sn
        ON dsf.permno = sn.permno
        AND dsf.date BETWEEN sn.namedt AND sn.nameenddt
    WHERE sn.ticker IN ({tickers})
        AND dsf.date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
        AND sn.shrcd IN (10, 11)
    """)
    
    # Get log-returns
    df['log_ret'] = np.log1p(df['ret'])
    
    # Drop NAs
    df = df[df['ret'].notna()].copy()
    
    # Set a dictionary of dataframes per ticker
    ret = df[['ticker', 'permno', 'hsiccd', 'shrcd','date', 'prc', 'ret', 'log_ret', 'vol']].copy()
    ret['date'] = pd.to_datetime(ret['date'])
    ret['year'] = ret['date'].dt.year
    ret_tickers = {ticker.upper(): ret[ret['ticker'] == ticker] for ticker in tickers_array}
    
    return ret_tickers


# -------------------------------------------------------------------------------------------

def rf_data(start_year, end_year):
    """
    Return dataframe with daily Fama-French risk-free rate.
    
    Given:
    - start_year: starting year
    - end_year  : ending year
    """
    # Call risk-free dataframe
    ff = db.raw_sql(f"""
    SELECT date, rf
    FROM ff_all.factors_daily
    WHERE date BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
    """)
    
    # Reinforce panda datetime
    ff['date'] = pd.to_datetime(ff['date'])

    # Create year column
    ff['year'] = ff['date'].dt.year
    
    return ff

# -------------------------------------------------------------------------------------------

def crsp_dividends(tickers_array, start_year, end_year):
    """
    Return cash dividends per (permno, ex-date) from stkdistributions,
    restricted to the given tickers and date range.
    
    Given:
    - tickers_array : array with tickers
    - start_year    : start of year
    - end_year      : end of year
    """
    # Convert array 
    tickers = np.sort(tickers_array)
    tickers = ', '.join(f"'{t}'" for t in tickers)

    # Dividends (cash only) 
    q = db.raw_sql(f"""
        SELECT d.permno,
               d.disexdt, d.disdeclaredt, d.dispaydt,
               d.disdivamt, d.distype
        FROM crsp.stkdistributions AS d
        WHERE d.disexdt BETWEEN '{start_year}-01-01' AND '{end_year}-12-31'
    """)
    q.rename(columns={
        'disexdt':'ex_date',
        'disdeclaredt':'decl_date',
        'dispaydt':'pay_date',
        'disdivamt':'div_per_share'
    }, inplace=True)
    
    # Keep cash dividends
    q = q[q['distype'] == 'CD'].copy()

    # Stocknames: bring in ticker with proper date windowing
    names = db.raw_sql(f"""
        SELECT sn.permno, sn.ticker, sn.namedt, sn.nameenddt
        FROM crsp_a_stock.stocknames AS sn
        WHERE sn.ticker IN ({tickers})
    """)
    
    # Merge dividends to names on permno and ex_date within [namedt, nameenddt]
    q['ex_date'] = pd.to_datetime(q['ex_date'])
    names['namedt'] = pd.to_datetime(names['namedt'])
    names['nameenddt'] = pd.to_datetime(names['nameenddt'])

    # Expand then filter to the correct name interval
    merged = q.merge(names, on='permno', how='left')
    in_window = (merged['ex_date'] >= merged['namedt']) & (merged['ex_date'] <= merged['nameenddt'])
    merged = merged.loc[in_window].copy()
    
    # Year of payment
    merged['ex_year'] = pd.to_datetime(merged['ex_date']).dt.year

    # Final columns
    out = merged[['ticker','permno','ex_date','decl_date','pay_date','div_per_share','distype','ex_year']].drop_duplicates()

    return out

# -------------------------------------------------------------------------------------------

def options_data(year, 
                 min_stock_vol=50000, 
                 min_stock_price=5, 
                 min_option_vol=100):
    """
    Returns a daily dataframe with Option Metrics data by linking 
    CRSP and Oprions Metrics identifiers for a given year.
    
    Given:
    - year            : year of the observations
    - min_stock_vol   >= volume of the stock (underlying asset)
    - min_stock_price >= price of the stock (underlying asset)
    - min_option_vol  >= volume of the option
    """
    # ---- 0. Year range --------------------------------------------------
    start = f'{year}-01-01'
    end   = f'{year}-12-31'

    # ---- 1. Identify PERMNOs in CRSP ----------------------------------------
    crsp_sql = f"""
        SELECT dsf.hsiccd, dsf.permno, sn.ticker, sn.shrcd,
            dsf.date, dsf.prc, dsf.vol
        FROM crsp_a_stock.dsf AS dsf
        LEFT JOIN crsp_a_stock.stocknames AS sn
            ON dsf.permno = sn.permno
            AND dsf.date BETWEEN sn.namedt AND sn.nameenddt
        WHERE (dsf.hsiccd BETWEEN 1300 AND 1399 OR dsf.hsiccd = 2911)  -- Oil sector stocks
            AND sn.shrcd IN (10, 11) -- Common stocks
            AND dsf.vol >= '{min_stock_vol}'
            AND dsf.prc > '{min_stock_price}'
            AND dsf.date BETWEEN '{start}' AND '{end}'
    """
    stocks           = db.raw_sql(crsp_sql)
    
    # Obtain PERMNOs
    permno_to_ticker = stocks[['permno', 'ticker']].drop_duplicates(subset='permno')
    
    # Store 'oil' PERMNOs in an array
    oil_permnos      = permno_to_ticker['permno'].unique()
    
    # Safeguard for empty list
    if len(oil_permnos) == 0:
        return pd.DataFrame()

    # ---- 2. Map PERMNOs to SECID ----------------------------------------
    # Set SQL input
    permnos_str = ", ".join(f"'{p}'" for p in oil_permnos)
    
    # Obtain link between PERMNO and SECID
    link = db.raw_sql(f"""
        SELECT permno, secid, sdate, edate
        FROM wrdsapps.opcrsphist
    """)
    
    # Sort values
    link_best = (
    link.sort_values(['permno', 'sdate'])
    .drop_duplicates('permno')
    )
    
    # Merge PERMNOs to filter for desired PERMNOs
    oil_link = (pd.DataFrame({'permno':oil_permnos})
           .merge(link_best, on='permno', how='inner'))
    
    # Merge for tickers
    oil_link = oil_link.merge(permno_to_ticker, on='permno', how='left')
    
    # Store list of SECIDs
    secid_list = oil_link['secid'].unique().tolist()
    # Safeguard for empty list
    if not secid_list:
        return pd.DataFrame()
    
    # ---- 3. Options data from Options Metrics ---------------------------
    # Set SQL input with SECID
    secid_str = ", ".join(f"'{s}'" for s in secid_list)
    
    # Call option data by year (e.g. opprcd2010, opprcd2011, ...)
    opt_tbl   = f"optionm.opprcd{year}"
    option_sql = f"""
        SELECT secid, date, symbol, exdate, cp_flag, ss_flag,
               volume, optionid, best_bid, best_offer,
               impl_volatility, strike_price, open_interest
        FROM {opt_tbl}
        WHERE secid IN ({secid_str})
          AND volume >= {min_option_vol}
    """
    options = db.raw_sql(option_sql)
    # Safeguard for empty dataframe
    if options.empty:
        return pd.DataFrame()
    
    # Create ticker column
    options['ticker'] = options['symbol'].apply(
        lambda s: s.split('.')[0] if '.' in s else s.split()[0]
    )
    
    # Select only options for companies in the oil sector
    options = options[options['ticker'].isin(oil_link['ticker'])]
    # Convert strike prices
    options['strike_price'] /= 1000
    # Calculate option prices (mid price)
    options['opt_price']     = (options['best_bid'] + options['best_offer'])/2
    # Calculate maturity days
    options['maturity_days'] = (
        pd.to_datetime(options['exdate']) - pd.to_datetime(options['date'])
    ).dt.days
    # Sort by ticker
    options = options.sort_values(by='ticker')
    
    # ---- 4. Underlying spot prices --------------------------------------
    # Set SQL input 
    tic_str = ", ".join(f"'{t}'" for t in options['ticker'].unique())
    
    # Get stock prices data from CRSP
    spot_sql = f"""
        SELECT sn.ticker, dsf.date, dsf.prc AS spot_price
        FROM crsp_a_stock.dsf AS dsf
        LEFT JOIN crsp_a_stock.stocknames AS sn
               ON dsf.permno = sn.permno
              AND dsf.date BETWEEN sn.namedt AND sn.nameenddt
        WHERE sn.ticker IN ({tic_str})
          AND dsf.date BETWEEN '{start}' AND '{end}'
    """
    spot = db.raw_sql(spot_sql)
    
    # Merge spot price data with options data
    options = options.merge(spot, on=['ticker', 'date'], how='left')
    
    # Calculate moneyness
    options['moneyness'] = options['strike_price'] / options['spot_price']
    # Calculate log-moneyness
    options['log_moneyness'] = np.log(options['moneyness'])
    
    return options

# -------------------------------------------------------------------------------------------

def filter_data(options_df, 
                ticker, 
                min_maturity=7,
                max_maturity=365,
                min_strikes=5
               ):
    """
    Returns a filtered dataframe with specified
    minimum time-to-maturiy, minimum number of strike prices,
    and arbitrage-free options.
    
    Given:
    - options_df    : dataframe to be filtered
    - ticker        : specific ticker symbol (string)
    - min_maturity  : minimum time-to-maturity in days
    - min_strikes   : minimum number of strikes per date
    """
    # Ticker
    df = options_df[options_df["ticker"] == ticker].copy()
    
    # Drop NAs
    df = df.dropna(subset=['date', 'strike_price', 'maturity_days'])
    
    # Ensure dates in strike_options are datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Keep only contracts with enough time to maturity 
    df = df[(df['maturity_days'] >= min_maturity) & (df['maturity_days'] <= max_maturity)].copy()
    
    # Keep only options with enough strike prices
    strike_counts = (
        df.groupby(['date', 'maturity_days'])['strike_price'].nunique().reset_index(name='n_strikes')
        
    )
    
    valid_combos = strike_counts[strike_counts['n_strikes'] >= min_strikes]
    
    df = df.merge(valid_combos[['date', 'maturity_days']], on=['date', 'maturity_days'], how='inner')
    
    # Get intrinsic values
    df["intrinsic_call"] = (df["spot_price"] - df["strike_price"]).clip(lower=0)
    df["intrinsic_put"]  = (df["strike_price"] - df["spot_price"]).clip(lower=0)
    
    # Call filter
    call_mask = (
        (df["cp_flag"] == "C") &
        (df["opt_price"] >= df["intrinsic_call"]) &
        (df["opt_price"] <= df["spot_price"])
    )
    
    # Put filter
    put_mask = (
        (df["cp_flag"] == "P") &
        (df["opt_price"] >= df["intrinsic_put"]) &
        (df["opt_price"] <= df["strike_price"])
    )
    
    # Get arbitrage-free dataframe
    df = df[call_mask | put_mask].copy()
    
    return df

# -------------------------------------------------------------------------------------------

def bs_price_vega(S, K, tau, r, q, sigma):
    """
    Returns vectorized Black–Scholes call prices and vega
    adjusted for a continuous dividend yield q.

    Given:
    - S     : underlying spot price
    - K     : strike price
    - tau   : time-to-maturity (in years)
    - r     : continuously-compounded risk-free rate
    - q     : continuously-compounded dividend yield
    - sigma : volatility
    """

    # Convert inputs to float arrays
    S     = np.asarray(S,     float)
    K     = np.asarray(K,     float)
    tau   = np.asarray(tau,   float)
    r     = np.asarray(r,     float)
    q     = np.asarray(q,     float)
    sigma = np.asarray(sigma, float)

    # Broadcast all inputs to a common shape
    shape = np.broadcast(S, K, tau, r, q, sigma).shape
    S, K, tau, r, q, sigma = [np.broadcast_to(x, shape) for x in (S, K, tau, r, q, sigma)]

    # Avoid division by zero (very small tau or sigma)
    sqrt_tau = np.sqrt(tau)
    eps      = 1e-16
    denom    = np.maximum(sigma * sqrt_tau, eps)

    # Compute d1 and d2 for BSM (note r - q inside d1)
    log_SK = np.log(S / K)
    d1 = (log_SK + ((r - q) + 0.5 * sigma**2) * tau) / denom
    d2 = d1 - sigma * sqrt_tau

    # Set discount factors
    disc_S = np.exp(-q * tau)  # present value adjustment for dividend yield
    disc_K = np.exp(-r * tau)  # present value adjustment for strike

    # BS call price
    price = S * disc_S * norm.cdf(d1) - K * disc_K * norm.cdf(d2)

    # Vega = dC/dsigma (IMPORTANT: uses pdf, not cdf)
    vega = S * disc_S * norm.pdf(d1) * sqrt_tau

    return price, vega

# -------------------------------------------------------------------------------------------

def bs_implied_vol(S, K, tau, r, q, C_mkt,
                   max_iter=20, tol=1e-8,
                   sigma_init=0.2,
                   sigma_min=1e-6, sigma_max=5.0):
    """
    Returns vectorized Black-Scholes implied volatility
    via Newton–Raphson with continuous dividend yield q.

    Given:
    - S     : underlying spot price
    - K     : strike price
    - tau   : time-to-maturity (in years)
    - r     : continuously-compounded risk-free rate
    - q     : continuously-compounded dividend yield
    - C_mkt : observed market call price
    """

    # Convert inputs to float arrays
    S     = np.asarray(S,     float)
    K     = np.asarray(K,     float)
    tau   = np.asarray(tau,   float)
    r     = np.asarray(r,     float)
    q     = np.asarray(q,     float)
    C_mkt = np.asarray(C_mkt, float)

    # Broadcast to a common shape
    shape = np.broadcast(S, K, tau, r, q, C_mkt).shape
    S, K, tau, r, q, C_mkt = [np.broadcast_to(x, shape) for x in (S, K, tau, r, q, C_mkt)]

    # Initialize sigma
    sigma = np.full(shape, sigma_init, float)

    # Set discount factors
    disc_S = np.exp(-q * tau)   # present value adjustment for dividend yield
    disc_K = np.exp(-r * tau)   # present value adjustment for strike

    # No-arbitrage lower bound
    intrinsic = np.maximum(S * disc_S - K * disc_K, 0.0)

    # Clamp market prices to feasible range to help Newton converge
    C_target = np.clip(C_mkt, intrinsic + 1e-8, S * disc_S - 1e-8)

    # Define small vega guard to avoid exploding Newton steps
    eps_vega = 1e-12

    for _ in range(max_iter):
        # Compute model price and vega at current sigma
        price, vega = bs_price_vega(S, K, tau, r, q, sigma)

        # Pricing error relative to the (clamped) target
        diff = price - C_target

        # Convergence mask (only update points not yet within tolerance)
        mask = np.abs(diff) > tol
        if not np.any(mask):
            break

        # Avoid dividing by tiny vega
        vega_safe = np.where(np.abs(vega) < eps_vega, eps_vega, vega)

        # Newton step: sigma_{new} = sigma - (price-target)/vega
        step = diff / vega_safe
        sigma = sigma - step * mask

        # Keep sigma within reasonable bounds
        sigma = np.clip(sigma, sigma_min, sigma_max)

    # Clean up non-finite and degenerate cases
    sigma[~np.isfinite(sigma)] = np.nan
    sigma[(C_mkt <= 0) | (tau <= 0)] = np.nan

    return sigma

# -------------------------------------------------------------------------------------------

def const_div_yield(df_div, df_stocks):
    """
    Returns a dataframe with a constant
    dividend yield q.

    Given:
    - df_div     : dataframe containing dividends data.
    - df_ticker  : dataframe containing stock prices.
    """

    # Aggregate total annual dividends per share
    div_annual = (
        df_div
        .groupby(['ticker', 'ex_year'], as_index=False)['div_per_share']
        .sum()
        .rename(columns={'div_per_share' : 'D_annual'})
    )

    # Compute yearly average stock prices
    price_annual = (
        df_stocks
        .groupby(['ticker', 'year'], as_index=False)['prc']
        .mean()
        .rename(columns={'prc' : 'S_avg'})
    )

    # Merge dividends and stock prices
    div_yield = (
        div_annual
        .merge(price_annual,
               left_on=['ticker', 'ex_year'],
               right_on=['ticker', 'year'],
               how='left')
    )

    # Set simple annual yield
    div_yield['div_yield_simple'] = div_yield['D_annual'] / div_yield['S_avg']

    # Convert to continuous yield
    div_yield['q'] = np.log1p(div_yield['div_yield_simple'])

    return div_yield

# -------------------------------------------------------------------------------------------

def starting_values(data,
                    tickers_array,
                    rolling_window=3,
                    k=4
                   ):
    """
    Returns a dataframe with parameter starting values
    per ticker to be estimated.

    Given:
    - data           : dataframe input
    - tickers_array  : array containing all tickers
    - rolling_window : number of years used to get values
    - k              : std. threshold to set jumps
    """

    # Copy dataframe
    df_all = data.copy()
    
    # Set list to store starting values
    results = []
    
    # Run loop to obtain starting values per ticker
    for t in tickers_array:
        df = df_all[df_all['ticker'] == t].copy()
        # Years list
        years = sorted(df['year'].unique())
        
        # Safeguard for empty data
        if len(years) == 0:
            continue
        
        # Run loop to estimate starting value per rolling window
        for i in range (0, len(years), rolling_window):
            start = years[i]
            end   = start + rolling_window - 1
            
            # Set window dataframe
            window_df = df[(df['year'] >= start) & (df['year'] <= end)].copy()
            
            # Safeguard for empty data
            if window_df.empty:
                continue
            
            # Get expected returns (log-returns mean)
            alpha     = window_df['log_ret'].mean()
            # Get Log-returns volatility
            sigma_hat = window_df['log_ret'].std(ddof=1)
            # Safeguard if sigma is too tiny or zero
            if not np.isfinite(sigma_hat) or sigma_hat <= 0:
                sigma_hat = 1e-12
            
            # Define jumps
            window_df['z_score'] = (window_df['log_ret'] - alpha) / sigma_hat
            n_window = len(window_df)
            mask     = np.abs(window_df['z_score']) > k
            jumps    = window_df[mask]
            n_jumps  = len(jumps)
            no_jumps = window_df[~mask]
            
            # Get Log-returns volatility with no jumps
            if not no_jumps.empty:
                sigma = (no_jumps['log_ret'].std(ddof=1)) * np.sqrt(252)
            else:
                sigma = sigma_hat * np.sqrt(252)
    
            if not jumps.empty:
                # Jump arrival rate
                lam   = (n_jumps / n_window) * 252
                # Jump mean
                mu    = jumps['log_ret'].mean()
                # Jump volatility
                delta = jumps['log_ret'].std(ddof=0)
                if not np.isfinite(delta) or delta  <= 0:
                    delta = 1e-8
            else:
                # Return NaN for no jumps
                lam    = 1e-8
                mu     = 1e-8
                delta  = 1e-8

            # Store results
            results.append({
                'ticker'       : t,
                'start_year'   : start,
                'end_year'     : end,
                'imp_vol'      : sigma,
                'jump_arrival' : lam,
                'jump_mean'    : mu,
                'jump_vol'     : delta

            })

    return pd.DataFrame(results)

# -------------------------------------------------------------------------------------------

def merton_call_vec(S, K, tau, r, q, sigma, lam, mu, delta, n_max=10, tol=1e-12):
    """
    Returns vectorized jump-diffusion (Merton) call prices
    adjusted for a continuous dividend yield q.

    Given:
    - S      : underlying spot price
    - K      : strike price
    - tau    : time-to-maturity (in years)
    - r      : continuously-compounded risk-free rate
    - q      : continuously-compounded dividend yield
    - sigma  : diffusion volatility
    - lam    : jump arrival rate (per year)
    - mu     : log-jump mean (per jump)
    - delta  : log-jump volatility (per jump)
    - n_max  : maximum Poisson term
    - tol    : optional early-stop tolerance on Poisson tail mass
    """

    # Convert inputs to float arrays
    S     = np.asarray(S,     float)
    K     = np.asarray(K,     float)
    tau   = np.asarray(tau,   float)
    r     = np.asarray(r,     float)
    q     = np.asarray(q,     float)
    sigma = np.asarray(sigma, float)
    lam   = np.asarray(lam,   float)
    mu    = np.asarray(mu,    float)
    delta = np.asarray(delta, float)

    # Broadcast all inputs to a common shape
    shape = np.broadcast(S, K, tau, r, q, sigma, lam, mu, delta).shape
    S, K, tau, r, q, sigma, lam, mu, delta = [
        np.broadcast_to(x, shape) for x in (S, K, tau, r, q, sigma, lam, mu, delta)
    ]

    # Numerical safeguards
    eps        = 1e-16
    tau_safe   = np.maximum(tau, eps)
    sigma_safe = np.maximum(sigma, eps)
    delta_safe = np.maximum(delta, 0.0)

    # Jump moments
    EJ    = np.exp(mu + 0.5 * delta_safe**2)   # E[J]
    kappa = EJ - 1.0                           # E[J - 1]

    # Poisson mean over [t, T]
    m = lam * tau_safe

    # Dividend discount factor for the stock leg
    disc_S = np.exp(-q * tau_safe)

    # Initialize mixture price
    price = np.zeros(shape, dtype=float)

    # Initial Poisson probability p_0
    p_n = np.exp(-m)
    cdf_mass = p_n.copy()

    # Mixture sum
    for n in range(0, int(n_max) + 1):
        # Effective volatility conditional on n jumps
        sigma_n2 = sigma_safe**2 + (n * delta_safe**2) / tau_safe
        sigma_n  = np.sqrt(np.maximum(sigma_n2, eps))

        # Effective rate inside the BS component
        r_n = (r - lam * kappa) + (n * (mu + 0.5 * delta_safe**2)) / tau_safe

        # ----- Dividend-adjusted Black–Scholes component -----

        sqrt_tau = np.sqrt(tau_safe)
        denom    = np.maximum(sigma_n * sqrt_tau, eps)

        log_SK = np.log(S / K)
        d1 = (log_SK + ((r_n - q) + 0.5 * sigma_n**2) * tau_safe) / denom
        d2 = d1 - sigma_n * sqrt_tau

        disc_K = np.exp(-r_n * tau_safe)
        bs_n   = S * disc_S * norm.cdf(d1) - K * disc_K * norm.cdf(d2)

        # Add weighted term
        price += p_n * bs_n

        # Update Poisson probability
        if n < n_max:
            p_n = p_n * (m / (n + 1.0))
            cdf_mass += p_n

            if np.all(1.0 - cdf_mass < tol):
                break

    return price


# -------------------------------------------------------------------------------------------

# Keep a canonical order for packing/unpacking vectors
PARAM_ORDER = ('sigma','lam','mu','delta')

def build_objective(S, K, tau, r, q, C_mkt, *, fixed=None, weights=None, n_max=20):
    """
    Returns residuals of modeled prices, given a dict of fixed params.
    
    Given:
    - S      : underlying asset price
    - K      : strike price
    - tau    : time to maturity
    - r      : risk-free rate
    - C_mkt  : option market price
    - fixed  : dict like {'sigma':0.2, 'mu':-0.05} to keep those pinned
    - weights: optional array (same shape as K/tau/C_mkt) for WLS
    """
    # Define fixed variables
    fixed = fixed or {}
    free_keys = [k for k in PARAM_ORDER if k not in fixed]

    # Broadcast weights -> 1 by default
    C_mkt = np.asarray(C_mkt, float)
    if weights is None:
        w = np.ones_like(C_mkt, dtype=float)
    else:
        w = np.broadcast_to(np.asarray(weights, float), C_mkt.shape)
    
    # Define which parameters are fixed and which parameters are to be estimated
    def unpack(x):
        # Turn x (vector of free params) into a full dict {sigma, lam, mu, delta}
        params = dict(fixed)
        params.update({k: float(v) for k, v in zip(free_keys, x)})
        return params
    
    # Calculate the residuals between Merton and market option prices
    def residuals(x):
        p = unpack(x)
        # Simple guards to keep solver in a valid region
        if p['sigma'] <= 0 or p['lam'] < 0 or p['delta'] < 0:
            return 1e6 * np.ones_like(C_mkt).ravel()

        C_mod = merton_call_vec(
            S, K, tau, r, q,
            p['sigma'], p['lam'], p['mu'], p['delta'],
            n_max=n_max
        )
        return (w * (C_mod - C_mkt)).ravel()

    return residuals, free_keys

# -------------------------------------------------------------------------------------------

def default_bounds(free_keys):
    """
    Returns bounds per parameter:
    - sigma: strictly positive and realistically < 1 (100% annual vol)
    - lam:   strictly positive (no zero jump rate) but not absurdly large
    - mu:    unbounded (allow both negative and positive jump means)
    - delta: strictly positive and realistically < 1 (jump vol per jump)
    """
    # Set list to store bounds
    lb, ub = [], []
    # Run loop per variable
    for k in free_keys:
        if k == 'sigma':  # diffusion volatility
            lb.append(1e-8)
            ub.append(1.0)
        if k == 'lam':    # jump arrival rate
            lb.append(1e-8)
            ub.append(10.0)
        if k == 'mu':     # jump mean (log space)
            lb.append(-np.inf)   # unbounded
            ub.append(np.inf)
        if k == 'delta':  # jump volatility
            lb.append(1e-8)
            ub.append(1.0)
    return (np.array(lb, float), np.array(ub, float))

# -------------------------------------------------------------------------------------------

def calibrate_merton(S, K, tau, r, q, C_mkt, *,
                     fixed=None, x0=None, bounds=None, weights=None,
                     n_max=20, method='trf', tol=1e-10, max_nfev=5000, verbose=1):
    """
    Generic wrapper. Provide:
      - fixed : dict of pinned params
      - x0    : starting values vector for FREE params (same order as returned free_keys)
      - bounds: (lb, ub) arrays for the FREE params (same order)
    
    Returns: (full_params_dict, scipy_result, free_keys)
    """
    # Get residuals
    residuals, free_keys = build_objective(S, K, tau, r, q, C_mkt,
                                           fixed=fixed, weights=weights, n_max=n_max)
    # Define safeguard
    if x0 is None:
        raise ValueError("Provide x0 for the free parameters in order: " + str(free_keys))
    
    # Define parameter bounds default
    if bounds is None:
        bounds = default_bounds(free_keys)

    # Run least square regression
    res = least_squares(residuals, x0=np.array(x0, float), bounds=bounds, method=method,
                        xtol=tol, ftol=tol, gtol=tol, max_nfev=max_nfev, verbose=verbose)

    # Stitch back full parameter dict
    theta = dict(fixed or {})
    theta.update({k: v for k, v in zip(free_keys, res.x)})
    return theta, res, free_keys

# -------------------------------------------------------------------------------------------

def prepare_inputs(by_maturity, *, D=252):
    """
    Returns vectorized market inputs (S, K, tau, r, q, C_mkt) 
    to be used in the price estimation.
    
    Given:
    - by_maturity : dataframe input
    - D           : number of trading days
    """
    
    # Convert underlying spot prices to array
    S       = by_maturity['spot_price'].to_numpy(float)
    # Convert strike price to array
    K       = by_maturity['strike_price'].to_numpy(float)
    # Convert time-to-expiration to array in trading years
    tau     = (by_maturity['maturity_days'].to_numpy(float) / D).astype(float)  # in years
    
    # Convert risk-free rate to array
    r_daily = by_maturity['rf'].to_numpy(float)
    # Convert risk-free rate in log-yearly terms
    r       = D * np.log1p(r_daily)

    # Convert constant dividend yield to array
    q = by_maturity['q'].to_numpy(float)
    
    # Convert option market price to array
    C_mkt   = by_maturity['opt_price'].to_numpy(float)
    
    return S, K, tau, r, q, C_mkt

# -------------------------------------------------------------------------------------------