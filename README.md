# eleven Hackathon – Product Recommendation (Case #2: “The Next Purchase”)

## Project description
We build a **personalized, stock-aware recommender system** for a sports retail company.
Using historical transactions and customer/product metadata, the system outputs **Top-K product recommendations per client** (products a client is most likely to buy next), suitable for marketing activation and filtered to recommend **only in-stock products** (by country when possible).

## Data (CSV files)
Place these files in `data/raw/`:

- `transactions.csv`  
  Historical purchases with (at least): `ClientID`, `ProductID`, `SaleTransactionDate`, `StoreID`, `Quantity`, `SalesNetAmountEuro`
- `clients.csv`  
  Customer attributes: segment, country, opt-ins, gender, age, etc.
- `products.csv`  
  Product catalog metadata: category / family / universe, etc.
- `stores.csv`  
  Store metadata: `StoreID` → `StoreCountry`
- `stocks.csv`  
  Available stock by country/product: `StoreCountry`, `ProductID`, `Quantity`

## Deliverables
We will produce:
1) **Recommendation output**: for each client, a ranked **Top-K list of ProductIDs**, applying stock constraints.
2) **EDA & insights**: plots/tables explaining customer behavior, sparsity/cold-start, country/segment differences, stock coverage, and “next purchase” patterns.
3) **Evaluation**: offline ranking metrics such as **Recall@K / MAP@K / NDCG@K**, with breakdowns by segment/country/cold-start.
4) **Demo-ready component**: a script or lightweight app that shows recommendations for a selected client with brief explanation.

---

## Setup

### Windows (recommended: PowerShell)
```powershell
# Create venv
py -m venv .venv

# Allow running the activate script (run once per machine)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate venv
& .\.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Train cold-start model (clients with 0 previous purchases)
# Run once to generate the model artifacts used by the demo UI
python -m src.newby_reco.reco_refacto --train

# Train repeat-customer model (clients with previous purchases)
python src/train.py
```

### Windows (Git Bash)
```bash
# Create venv
python -m venv .venv

# Activate venv
source .venv/bin/activate

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Train cold-start model (clients with 0 previous purchases)
# Run once to generate the model artifacts used by the demo UI
python -m src.newby_reco.reco_refacto --train

# Train repeat-customer model (clients with previous purchases)
python src/train.py
```

### MacOS / Linux
```bash
# Create venv
python3 -m venv .venv

# Activate venv
source .venv/bin/activate

# Install dependencies
python3 -m pip install --upgrade pip
pip install -r requirements.txt

# Train cold-start model (clients with 0 previous purchases)
# Run once to generate the model artifacts used by the demo UI
python3 -m src.newby_reco.reco_refacto --train

# Train repeat-customer model (clients with previous purchases)
python3 src/train.py
```

## Run the EDA App (Streamlit)

This repo includes an internal Streamlit app for exploring the dataset and building the business case for a stock-aware recommendation system.

From the repo root (after setup and with CSVs placed in `data/raw/`):

```bash
streamlit run app_eda/Home.py
```


## Run the Client Demo App (Streamlit)

From the repo root (after setup and with CSVs placed in `data/raw/`):

```bash
streamlit run app_demo/Home.py
```
