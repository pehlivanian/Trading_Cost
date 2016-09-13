# QB

# TransactionCost Model

This was a 2-week project for QB, and includes calibrations on
{STROBE, BOLT} trades from their CME futures trade history. All trades
are agency trades executed on CME Group.

## Model basics

The main model object CMECostModel() encapsulates the data and fitting
procedures. Reporters are visitors of the main model object CMECostModel. Example run in the driver.py script.

We attempt to explain trading cost using pre-trade features.
Specifically:

trading cost (bps)	= 1e4 * sidesign * (avg exec. price - arrival_price)/arrival_price

feature_0              	=  1 (const)

feature_1             	=  trade_duration * pct_of_volume * sqrt(name_variance * fraction_of_day)

feature_2             	=  bid-ask spread measure

The model chosen was a power-law in the features, as this was the most
parsimonious and accurate specification of the well-recognized
concavity of trading cost in duration pov.




---

Sample run to produce preliminary reports:

python driver.py
