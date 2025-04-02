import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
import joblib
import matplotlib.pyplot as plt

class DataDriftDetector:
    def __init__(self, production_data_path, training_data_path):
        self.prod_data = pd.read_csv(production_data_path)
        self.train_data = pd.read_csv(training_data_path)
        self.features = ['duration', 'call_time', 'region', 'product']
        
    def detect_drift(self, threshold=0.05):
        results = {}
        for feature in self.features:
            if self.prod_data[feature].dtype == 'float64':
                # Test KS pour les variables continues
                stat, p = ks_2samp(
                    self.train_data[feature],
                    self.prod_data[feature]
                )
            else:
                # Test Chi² pour les variables catégorielles
                contingency = pd.crosstab(
                    self.train_data[feature],
                    self.prod_data[feature]
                )
                _, p, _, _ = chi2_contingency(contingency)
                
            results[feature] = {
                'p_value': p,
                'drift_detected': p < threshold
            }
        
        self.generate_report(results)
        return results
    
    def generate_report(self, results):
        plt.figure(figsize=(10, 6))
        pd.DataFrame(results).T['p_value'].plot(kind='bar', color='skyblue')
        plt.axhline(y=0.05, color='r', linestyle='--')
        plt.title('Data Drift Analysis Report')
        plt.ylabel('p-value')
        plt.savefig('reports/drift_report.png')