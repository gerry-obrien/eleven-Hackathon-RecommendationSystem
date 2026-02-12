import pandas as pd
import numpy as np
import os
import pickle
from tqdm import tqdm
from engine import SegmentRecommender

DATA_PATH = 'data/raw'
MODEL_PATH = 'models/'
OUTPUT_PATH = 'output/'
SAMPLE_SIZE = 500
K_RECS = 5

os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(OUTPUT_PATH, exist_ok=True)

def load_data():
    print("⏳ Loading CSVs...")
    transactions = pd.read_csv(os.path.join(DATA_PATH, 'transactions.csv'))
    clients = pd.read_csv(os.path.join(DATA_PATH, 'clients.csv'))
    products = pd.read_csv(os.path.join(DATA_PATH, 'products.csv'))
    
    try:
        stocks = pd.read_csv(os.path.join(DATA_PATH, 'stocks.csv'))
        stocks['ProductID'] = stocks['ProductID'].astype(str)
        stock_lookup = stocks.set_index('ProductID')['Quantity'].to_dict()
    except FileNotFoundError:
        stock_lookup = {}

    transactions['ClientID'] = transactions['ClientID'].astype(str)
    transactions['ProductID'] = transactions['ProductID'].astype(str)
    clients['ClientID'] = clients['ClientID'].astype(str)
    products['ProductID'] = products['ProductID'].astype(str)
    transactions['SaleTransactionDate'] = pd.to_datetime(transactions['SaleTransactionDate'])

    products['CustomName'] = (
        '(' + products['Universe'].astype(str) + ') ' + 
        products['Category'].astype(str) + ' ' + 
        products['FamilyLevel2'].astype(str)
    )
    name_lookup = products.set_index('ProductID')['CustomName'].to_dict()
    
    return transactions, clients, products, name_lookup, stock_lookup

def create_time_split(transactions, clients):
    print("⏳ Creating Train/Test Split...")
    transactions = transactions.sort_values(['ClientID', 'SaleTransactionDate'])
    
    last_dates = transactions.groupby('ClientID')['SaleTransactionDate'].max().reset_index()
    last_dates.columns = ['ClientID', 'LastDate']
    
    transactions_dates = transactions.merge(last_dates, on='ClientID', how='left')
    
    test_df = transactions_dates[transactions_dates['SaleTransactionDate'] == transactions_dates['LastDate']]
    train_df = transactions_dates[transactions_dates['SaleTransactionDate'] < transactions_dates['LastDate']]
    
    valid_test_clients = test_df[test_df['ClientID'].isin(train_df['ClientID'].unique())]
    train_df_valid = train_df[train_df['ClientID'].isin(valid_test_clients['ClientID'].unique())]
    
    train_merged = train_df_valid.merge(clients[['ClientID', 'ClientCountry', 'ClientSegment']], on='ClientID', how='left')
    test_merged = valid_test_clients.merge(clients[['ClientID', 'ClientCountry', 'ClientSegment']], on='ClientID', how='left')
    
    return train_merged.dropna(subset=['ClientSegment']), test_merged.dropna(subset=['ClientSegment'])

def train_and_evaluate():
    transactions, clients, products, name_lookup, stock_lookup = load_data()
    train_df, test_df = create_time_split(transactions, clients)
    
    print("⏳ Training Segment Models...")
    models = {}
    groups = train_df.groupby(['ClientCountry', 'ClientSegment'])

    for (country, segment), subset in groups:
        if len(subset) > 5:
            models[(country, segment)] = SegmentRecommender(subset, name_lookup, stock_lookup)

    print("\n⏳ Running Evaluation...")
    all_test_ids = test_df['ClientID'].unique()
    test_client_ids = np.random.choice(all_test_ids, min(SAMPLE_SIZE, len(all_test_ids)), replace=False)
        
    hits = 0
    total_cases = 0
    results_log = []

    for client_id in tqdm(test_client_ids):
        actual_products = test_df[test_df['ClientID'] == client_id]['ProductID'].unique()
        try:
            client_meta = train_df[train_df['ClientID'] == client_id].iloc[0]
            key = (client_meta['ClientCountry'], client_meta['ClientSegment'])
            
            if key in models:
                recs_df = models[key].recommend(client_id, n_recs=K_RECS)
                if not recs_df.empty:
                    predicted_products = recs_df['ProductID'].values
                    is_hit = 1 if len(set(actual_products).intersection(set(predicted_products))) > 0 else 0
                    hits += is_hit
                    total_cases += 1
                    results_log.append({
                        'ClientID': client_id, 'Hit': is_hit,
                        'Actual': str(list(actual_products)), 'Predicted': str(list(predicted_products))
                    })
        except Exception:
            continue

    accuracy = (hits / total_cases) * 100 if total_cases > 0 else 0
    summary = f"--- EVALUATION REPORT ---\nSample Size: {total_cases}\nHit Rate @ {K_RECS}: {accuracy:.2f}%\n"
    print(summary)
    
    with open(os.path.join(OUTPUT_PATH, 'evaluation_summary.txt'), 'w') as f:
        f.write(summary)
    pd.DataFrame(results_log).to_csv(os.path.join(OUTPUT_PATH, 'evaluation_logs.csv'), index=False)

    print(f"✅ Saving Models to {MODEL_PATH}...")
    with open(os.path.join(MODEL_PATH, 'segment_models.pkl'), 'wb') as f:
        pickle.dump(models, f)
    client_map = clients.set_index('ClientID')[['ClientCountry', 'ClientSegment']].to_dict('index')
    with open(os.path.join(MODEL_PATH, 'client_map.pkl'), 'wb') as f:
        pickle.dump(client_map, f)
    print("🎉 Done! Models and Evaluation saved.")

if __name__ == "__main__":
    train_and_evaluate()
