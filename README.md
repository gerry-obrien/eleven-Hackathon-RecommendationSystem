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

## Static Slide Plot Export

Generate the slide-ready heavy-tail chart from raw data:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py
```

Default output:

```text
plots/heavy_tail_transactions_per_client_clip50.png
```

Optional overrides:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --clip-upper 34 --color "#5c8bc3" --dpi 300
```

To match Streamlit-like bin density, keep `--nbins 50` (default):

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --clip-upper 50 --nbins 50
```

For non-16:9 placeholders in slides, you can tune figure dimensions:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --clip-upper 34 --nbins 50 --fig-width 12 --fig-height 8 --dpi 220 --out-name heavy_tail_transactions_per_client_clip34_slide_fit.png
```

Saved preset for the 1.45 slide fit:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --clip-upper 34 --nbins 50 --preset slide_fit_145
```

Deeper strategy-focused chart for "Distribution of past purchases":

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --chart past_purchase_strategy --clip-upper 34 --nbins 50 --preset slide_fit_145
```

Comprehensive slide dimension (large center panel, ~1.44 aspect):

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --chart past_purchase_strategy --clip-upper 34 --nbins 50 --fig-width 13 --fig-height 9 --dpi 220 --out-name past_purchase_strategy_clip34_comprehensive_slide.png
```

Wider center-panel aspect (for short/tall-constrained frames):

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --chart past_purchase_strategy --clip-upper 34 --nbins 50 --preset comprehensive_wide_220
```

Comprehensive Country x Segment executive slide:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --chart country_segment_comprehensive --preset country_segment_exec_wide
```

Default output artifact:

```text
plots/country_segment_comprehensive_exec.png
```

Stretched aspect ratio for wide/short slide placeholders:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --chart country_segment_comprehensive --fig-width 20 --fig-height 8.4 --dpi 220 --out-name country_segment_comprehensive_exec_stretched.png
```

Stock actionability executive slide:

```powershell
.venv\Scripts\python.exe scripts/export_slide_plots.py --chart stock_actionability --preset stock_actionability_exec
```

Default output artifact:

```text
plots/stock_actionability_exec.png
```

## Run the Client Demo App (Streamlit)

From the repo root (after setup and with CSVs placed in `data/raw/`):

```bash
streamlit run app_demo/Home.py
```
