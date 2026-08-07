# Data Documentation

## Data Availability

This project uses security-level financial data obtained through Wharton Research Data Services (WRDS).

The underlying CRSP, Compustat, and CRSP/Compustat Merged observations are subject to institutional licensing restrictions. They therefore cannot be redistributed through this public repository.

The repository does not contain raw or security-level processed datasets.

## Data Sources

### CRSP

The CRSP data used in this project include:

- Historical S&P 500 constituents
- Daily stock prices
- Daily total returns
- Trading volume
- Shares outstanding
- Market capitalization
- Official S&P 500 value-weighted benchmark returns
- Official S&P 500 equal-weighted benchmark returns

### Compustat North America

The Compustat annual fundamentals include:

- Book equity
- Earnings
- Total assets
- Revenue
- Profitability
- Debt
- Cash flow
- Capital expenditure
- Investment and accrual-related variables

### CRSP/Compustat Merged Database

The historical CCM link table is used to connect:

- Compustat `GVKEY`
- CRSP `PERMNO`
- CRSP `PERMCO`

Historical link start dates, end dates, link types, and link priorities are applied during the merging process.

### Kenneth R. French Data Library

Publicly available monthly factor data include:

- Market excess return
- SMB
- HML
- RMW
- CMA
- Momentum
- Risk-free rate

## Directory Structure

```text
data
├── raw
├── interim
├── processed
└── README.md
