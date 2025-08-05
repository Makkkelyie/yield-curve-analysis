import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve
from scipy.stats import norm
import QuantLib as ql
import math
from scipy.optimize import curve_fit
import yfinance as yf

df = pd.read_csv('/Users/kellymao93/Desktop/prices.csv')
df['time'] = pd.to_datetime(df['time'], unit='ms')
##(1)
#Find the most recent row where all US yields are non-zero
valid = df[df[['US02', 'US05', 'US10', 'US30']].ne(0).all(axis=1)]
latest = valid.iloc[-1]

#Set maturities and their corresponding yields
maturities = [2, 5, 10, 30]
yields = latest[['US02', 'US05', 'US10', 'US30']].values

#plot the yield curve
plt.plot(maturities, yields, marker='o')
plt.title('US Yield Curve - Most Recent Valid Date')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.show()

#Print the latest row for confirmation
print(latest[['time', 'US02', 'US05', 'US10', 'US30']])
##(2)
# Function to calculate bond price
def bond_price(yield_percent, n, face_value=100, coupon_rate=0.02):
    y = yield_percent / 100
    C = face_value * coupon_rate
    return sum([C / (1 + y)**t for t in range(1, n + 1)]) + face_value / (1 + y)**n

# Calculate and print bond prices for 2Y, 5Y, 10Y, 30Y
for name, n in zip(['US02', 'US05', 'US10', 'US30'], [2, 5, 10, 30]):
    price = bond_price(latest[name], n)
    print(f"{n}-year bond price (coupon 2%): ${price:.2f}")

##(3)
# plot the bond price curve
prices = [bond_price(latest[name], n) for name, n in zip(['US02', 'US05', 'US10', 'US30'], maturities)]
plt.plot(maturities, prices, marker='o', color='green')
plt.title('Bond Price Curve (Coupon 2%)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Price ($)')
plt.grid(True)
plt.show()

# Filter the most recent row with all yields present
required_columns = ['US02', 'US05', 'US10', 'US30',
                    'CA02', 'CA05', 'CA10', 'CA30',
                    'AU02', 'AU05', 'AU10', 'AU30']

# Drop rows where any required yield is 0
valid = df[df[required_columns].ne(0).all(axis=1)]

# Use the most recent valid row
latest = valid.iloc[-1]

# Define maturities
maturities = [2, 5, 10, 30]

# Extract yields
us_yields = latest[['US02', 'US05', 'US10', 'US30']].values
ca_yields = latest[['CA02', 'CA05', 'CA10', 'CA30']].values
au_yields = latest[['AU02', 'AU05', 'AU10', 'AU30']].values

# Plot yield curves
plt.plot(maturities, us_yields, marker='o', label='USA')
plt.plot(maturities, ca_yields, marker='o', label='Canada')
plt.plot(maturities, au_yields, marker='o', label='Australia')

# Format the chart
plt.title('Yield Curve Comparison (USA, Canada, Australia)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Yield (%)')
plt.grid(True)
plt.legend()
plt.show()

# Print out the yields for reference
print("USA Yields:", us_yields)
print("Canada Yields:", ca_yields)
print("Australia Yields:", au_yields)

##(4)
latest_yields = {
    'US02': 4.45,
    'US05': 4.00,
    'US10': 3.85,
    'US30': 3.60
}

# Function to calculate bond price
def bond_price(yield_percent, n, face_value=100, coupon_rate=0.02):
    y = yield_percent / 100
    C = face_value * coupon_rate
    return sum([C / (1 + y)**t for t in range(1, n + 1)]) + face_value / (1 + y)**n

# Function to calculate Macaulay Duration
def macaulay_duration(yield_percent, n, face_value=100, coupon_rate=0.02):
    y = yield_percent / 100
    C = face_value * coupon_rate
    P = bond_price(yield_percent, n, face_value, coupon_rate)
    duration = sum([t * C / (1 + y)**t for t in range(1, n + 1)]) + n * face_value / (1 + y)**n
    return duration / P

# Function to calculate Modified Duration
def modified_duration(yield_percent, n, face_value=100, coupon_rate=0.02):
    mac_dur = macaulay_duration(yield_percent, n, face_value, coupon_rate)
    y = yield_percent / 100
    return mac_dur / (1 + y)

# Function to calculate Convexity
def convexity(yield_percent, n, face_value=100, coupon_rate=0.02):
    y = yield_percent / 100
    C = face_value * coupon_rate
    P = bond_price(yield_percent, n, face_value, coupon_rate)
    convex = sum([t * (t + 1) * C / (1 + y)**(t + 2) for t in range(1, n + 1)]) + \
             n * (n + 1) * face_value / (1 + y)**(n + 2)
    return convex / P

# Build a DataFrame for results
maturities = [2, 5, 10, 30]
results = []
for name, n in zip(['US02', 'US05', 'US10', 'US30'], maturities):
    yld = latest_yields[name]
    results.append({
        "Maturity": n,
        "Yield (%)": yld,
        "Bond Price": bond_price(yld, n),
        "Macaulay Duration": macaulay_duration(yld, n),
        "Modified Duration": modified_duration(yld, n),
        "Convexity": convexity(yld, n)
    })

df_results = pd.DataFrame(results)
print(df_results.to_string())

# Plot Duration vs Maturity
plt.plot(df_results["Maturity"], df_results["Macaulay Duration"], marker='o', label="Macaulay Duration")
plt.plot(df_results["Maturity"], df_results["Modified Duration"], marker='x', label="Modified Duration")
plt.title("Duration vs. Maturity")
plt.xlabel("Maturity (Years)")
plt.ylabel("Duration")
plt.legend()
plt.grid(True)
plt.show()

##(5)Use Bootstrapping to Derive Spot Rates
# Bootstrap to find spot rates from par yield curve
def bootstrap_spot_rates(par_yields, maturities, face_value=100, coupon_rate=0.02):
    spot_rates = []

    for i, (y, n) in enumerate(zip(par_yields, maturities)):
        C = face_value * coupon_rate
        if i == 0:
            # For 1st maturity, spot rate equals par yield
            r = y / 100
        else:
            # Define the function to solve
            def price_eq(r):
                return sum([C / (1 + spot_rates[j])**(j + 1) for j in range(i)]) + \
                       (C + face_value) / (1 + r)**n - face_value
            # Solve for r
            r = fsolve(price_eq, y / 100)[0]
        spot_rates.append(r)
    return np.array(spot_rates) * 100  # Convert to %

# Get par yields from latest row
par_yields = np.array([latest_yields['US02'], latest_yields['US05'], latest_yields['US10'], latest_yields['US30']])
spot_rates = bootstrap_spot_rates(par_yields, maturities)

# Plot comparison
plt.plot(maturities, par_yields, marker='o', label='Par Yield Curve')
plt.plot(maturities, spot_rates, marker='x', label='Spot Rate Curve')
plt.title('Par Yield vs Spot Rate Curve (Bootstrapped)')
plt.xlabel('Maturity (Years)')
plt.ylabel('Rate (%)')
plt.legend()
plt.grid(True)
plt.show()

# Print spot rates for reference
print("Bootstrapped Spot Rates (%):")
for m, s in zip(maturities, spot_rates):
    print(f"{m}-year: {s:.3f}%")

##(6)Simulate Pricing a Swaption or Bond Option (Black Model)
# Set evaluation date
today = ql.Date.todaysDate()
ql.Settings.instance().evaluationDate = today

# Parameters
face_value = 100
coupon_rate = 0.02
yield_rate = 0.04
maturity_years = 5
volatility = 0.20  # 20%
option_expiry = today + ql.Period(6, ql.Months)  # Option matures in 6 months
strike_price = 98

# Construct the bond
calendar = ql.UnitedStates(ql.UnitedStates.GovernmentBond)
end_date = calendar.advance(today, ql.Period(maturity_years, ql.Years))
schedule = ql.Schedule(
    today,
    end_date,
    ql.Period(1, ql.Years),
    calendar,
    ql.Following,
    ql.Following,
    ql.DateGeneration.Backward,
    False
)

# Yield curve
fixed_rate_bond = ql.FixedRateBond(1, face_value, schedule, [coupon_rate], ql.ActualActual(ql.ActualActual.Bond))

spot_curve = ql.FlatForward(today, yield_rate, ql.ActualActual(ql.ActualActual.Bond), ql.Compounded, ql.Annual)

yield_curve_handle = ql.YieldTermStructureHandle(spot_curve)

# Bond engine
bond_engine = ql.DiscountingBondEngine(yield_curve_handle)
fixed_rate_bond.setPricingEngine(bond_engine)

# Forward value of bond (assume forward is current dirty price for simplicity)
forward_price = fixed_rate_bond.cleanPrice()

# Define the Black calculator
black_calc = ql.BlackCalculator(
    ql.PlainVanillaPayoff(ql.Option.Call, strike_price),
    forward_price,
    volatility * math.sqrt(0.5),  # √T with T = 0.5 years
    np.exp(-yield_rate * 0.5)
)

# Get the call option price
call_price = black_calc.value()
print(f"Call option price on the bond (QuantLib): ${call_price:.4f}")

##(7)Time-Series Analysis: Yield Curve Shifts
# Filter only valid rows (non-zero yields)
valid_df = df[df[['US02', 'US05', 'US10', 'US30']].ne(0).all(axis=1)]
# Keep a few recent dates
recent = valid_df.tail(5).copy()
recent['time'] = pd.to_datetime(recent['time'])
# Calculate daily yield changes (differences)
recent['US02_chg'] = recent['US02'].diff()
recent['US05_chg'] = recent['US05'].diff()
recent['US10_chg'] = recent['US10'].diff()
recent['US30_chg'] = recent['US30'].diff()

def classify_shift(row):
    changes = [row['US02_chg'], row['US05_chg'], row['US10_chg'], row['US30_chg']]
    if all(np.sign(c) == np.sign(changes[0]) for c in changes if not pd.isna(c)):
        if abs(changes[0] - changes[-1]) < 0.02:
            return "Parallel Shift"
        elif changes[-1] > changes[0]:
            return "Steepening"
        else:
            return "Flattening"
    return "Mixed"

recent['Shift Type'] = recent.apply(classify_shift, axis=1)
maturities = [2, 5, 10, 30]

plt.figure(figsize=(10, 6))
for i in range(len(recent)):
    plt.plot(
        maturities,
        recent.iloc[i][['US02', 'US05', 'US10', 'US30']],
        label=recent.iloc[i]['time'].strftime('%Y-%m-%d')
    )
plt.title("US Yield Curve Shifts Over Time")
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(True)
plt.show()

##(8)Fit Yield Curve with Nelson-Siegel Model
def nelson_siegel(t, beta0, beta1, beta2, tau):
    term1 = (1 - np.exp(-t / tau)) / (t / tau)
    term2 = term1 - np.exp(-t / tau)
    return beta0 + beta1 * term1 + beta2 * term2
# Maturities and corresponding yields
maturities = np.array([2, 5, 10, 30])
yields = np.array([latest_yields['US02'], latest_yields['US05'], latest_yields['US10'], latest_yields['US30']])
# Initial guess for [β0, β1, β2, τ]
initial_guess = [3.0, -1.0, 1.0, 1.0]

# Fit the Nelson-Siegel model
params, _ = curve_fit(nelson_siegel, maturities, yields, p0=initial_guess, maxfev=10000)
beta0, beta1, beta2, tau = params

print(f"Fitted Parameters:\nβ0 (Level): {beta0:.4f}, β1 (Slope): {beta1:.4f}, β2 (Curvature): {beta2:.4f}, τ: {tau:.4f}")
# Generate smooth curve
t_fit = np.linspace(0.5, 30, 100)
y_fit = nelson_siegel(t_fit, *params)

# Plot actual vs fitted
plt.figure(figsize=(10, 6))
plt.plot(maturities, yields, 'ro', label='Observed Yields')
plt.plot(t_fit, y_fit, 'b-', label='Nelson-Siegel Fit')
plt.title("Yield Curve Fitting - Nelson-Siegel Model")
plt.xlabel("Maturity (Years)")
plt.ylabel("Yield (%)")
plt.legend()
plt.grid(True)
plt.show()

##(9)
# Define tickers (can use treasury ETFs as yield proxies)
tickers = {
    "2Y": "SHY",     # 1–3 Yr Treasury Bond ETF
    "5Y": "IEI",     # 3–7 Yr Treasury Bond ETF
    "10Y": "IEF",    # 7–10 Yr Treasury Bond ETF
    "30Y": "TLT",    # 20+ Yr Treasury Bond ETF
}

# Fetch latest data
yield_data = {}
for key, ticker in tickers.items():
    etf = yf.Ticker(ticker)
    price = etf.history(period='1d')['Close'].iloc[-1]
    yield_data[key] = price

print("Latest Bond ETF Prices (as yield proxies):")
for k, v in yield_data.items():
    print(f"{k}: {v:.2f}")