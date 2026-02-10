from src.load_data import load_all
from src.evaluate import last_purchase_split, summarize_metrics
from src.recsys import popularity_recs, build_stock_map, filter_recs_by_stock

def main():
    data = load_all()
    train, test = last_purchase_split(data.transactions)

    # baseline recs
    recs_raw = popularity_recs(train, k=10)

    # evaluate raw
    truth = test.set_index("ClientID")["ProductID"]
    print("RAW:")
    print(summarize_metrics(recs_raw, truth, k=10).to_string(index=False))

    # stock filtering
    stock_map = build_stock_map(data.stocks)
    client_country = data.clients.set_index("ClientID")["ClientCountry"]
    recs_stock = filter_recs_by_stock(recs_raw, client_country, stock_map)

    print("\nSTOCK-FILTERED:")
    print(summarize_metrics(recs_stock, truth, k=10).to_string(index=False))

if __name__ == "__main__":
    main()
