import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfTransformer

class SegmentRecommender:
    def __init__(self, data_subset, name_lookup, stock_lookup):
        self.data = data_subset
        self.name_lookup = name_lookup
        self.stock_lookup = stock_lookup
        self.country = data_subset['ClientCountry'].iloc[0]
        self.segment = data_subset['ClientSegment'].iloc[0]
        
        # Mappings
        self.user_c = self.data['ClientID'].astype('category')
        self.product_c = self.data['ProductID'].astype('category')
        
        # Matrix Construction
        row_indices = self.user_c.cat.codes
        col_indices = self.product_c.cat.codes
        values = self.data['Quantity']
        
        self.raw_matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self.user_c.cat.categories), len(self.product_c.cat.categories))
        )
        
        # TF-IDF & Model
        self.tfidf = TfidfTransformer(norm='l2', use_idf=True)
        self.tfidf_matrix = self.tfidf.fit_transform(self.raw_matrix)
        
        k = min(10, self.tfidf_matrix.shape[0])
        self.model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=k, n_jobs=-1)
        self.model.fit(self.tfidf_matrix)

    def recommend(self, client_id, n_recs=5):
        client_id = str(client_id)
        if client_id not in self.user_c.cat.categories:
            return pd.DataFrame()

        # Get Index & Neighbors
        user_idx = self.user_c.cat.categories.get_loc(client_id)
        distances, indices = self.model.kneighbors(self.tfidf_matrix[user_idx])
        
        neighbor_indices = indices.flatten()
        similarities = (1 - distances.flatten()).reshape(-1, 1)
        
        neighbor_matrix = self.tfidf_matrix[neighbor_indices]
        rec_vector = neighbor_matrix.multiply(similarities).sum(axis=0).A.flatten()
        
        # Sort
        top_indices = rec_vector.argsort()[::-1][:n_recs]
        
        results = []
        for i in top_indices:
            score = rec_vector[i]
            if score > 0:
                pid = self.product_c.cat.categories[i]
                p_name = self.name_lookup.get(pid, "Unknown Product")
                p_stock = self.stock_lookup.get(pid, 0)
                
                results.append({
                    'ClientID': client_id,
                    'ProductID': pid,
                    'ProductName': p_name,
                    'Stock': p_stock,
                    'Score': round(score, 4),
                    'Segment': f"{self.country}-{self.segment}"
                })
        return pd.DataFrame(results)

    def explain_neighbors(self, client_id, k=5):
        client_id = str(client_id)
        if client_id not in self.user_c.cat.categories:
            return pd.DataFrame()
            
        user_idx = self.user_c.cat.categories.get_loc(client_id)
        distances, indices = self.model.kneighbors(self.tfidf_matrix[user_idx], n_neighbors=k+1)
        
        neighbor_indices = indices.flatten()
        neighbor_distances = distances.flatten()
        
        detailed_history = []
        
        # Skip index 0 (self)
        for i in range(1, len(neighbor_indices)):
            n_idx = neighbor_indices[i]
            n_real_id = self.user_c.cat.categories[n_idx]
            similarity = 1 - neighbor_distances[i]
            
            raw_row = self.raw_matrix[n_idx].toarray().flatten()
            purchased_indices = np.where(raw_row > 0)[0]
            
            for pid_idx in purchased_indices:
                real_pid = self.product_c.cat.categories[pid_idx]
                qty = raw_row[pid_idx]
                p_name = self.name_lookup.get(real_pid, "Unknown Product")
                
                detailed_history.append({
                    'NeighborRank': i,
                    'NeighborID': n_real_id,
                    'Similarity': round(similarity, 4),
                    'ProductID': real_pid,
                    'ProductName': p_name,
                    'Quantity': int(qty)
                })
                
        return pd.DataFrame(detailed_history)
