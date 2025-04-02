"""
Script pour générer des données réalistes de campagnes commerciales
Features:
- Génération de logs d'appels avec patterns réalistes
- Simulation de variations temporelles (saisonalité hebdo)
- Export en CSV et vérification de la qualité
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random
from typing import Dict, List
import logging
import argparse
from pathlib import Path

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, seed: int = 42):
        """Initialisation avec reproductibilité"""
        self.fake = Faker()
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration métier
        self.scripts = {
            'A': {'base_conversion': 0.25, 'time_impact': {9: -0.1, 14: 0.2}},
            'B': {'base_conversion': 0.35, 'time_impact': {10: 0.1, 15: 0.3}},
            'C': {'base_conversion': 0.15, 'time_impact': {16: 0.4}}
        }
        
        self.regions = ['North', 'South', 'East', 'West']
        self.products = ['SaaS', 'Hardware', 'Consulting']
        
        
        self.holidays = {
            '2024': ['2024-01-01', '2024-04-01', '2024-05-08', '2024-07-14'],
            '2023': ['2023-01-01', '2023-04-10', '2023-05-01', '2023-07-14']
        }
        
    def _generate_call_time(self, day: datetime) -> Dict:
        """Génère un timestamp réaliste avec variations jour/semaine"""
        hour = np.random.choice(
            [9, 10, 14, 15, 16], 
            p=[0.15, 0.25, 0.3, 0.2, 0.1]
        )
        
        # Effet weekend
        if day.weekday() >= 5:
            hour = max(10, hour - 1)  # Début plus tard le weekend
            
        return {
            'call_time': hour,
            'call_date': day.strftime('%Y-%m-%d'),
            'day_of_week': day.strftime('%A')
        }
    
    def _get_holiday_effect(self, date_str: str) -> float:
        """Calcule l'impact des jours fériés sur la conversion"""
        year = date_str[:4]
        if date_str in self.holidays.get(year, []):
            return -0.4  # -40% de conversion les jours fériés
        return 0.0
    
    def _determine_conversion(self, script: str, hour: int, region: str, date_str: str) -> int:
        """Calcule la probabilité de conversion avec effets combinés"""
        base_rate = self.scripts[script]['base_conversion']
        
        # Effet horaire
        hour_impact = self.scripts[script]['time_impact'].get(hour, 0)
        
        # Effet région (ex: East performe mieux)
        region_impact = 0.1 if region == 'East' else -0.05 if region == 'North' else 0
        
        # Bruit aléatoire
        noise = np.random.normal(0, 0.05)
        
        # Ajout effet jours fériers
        holiday_impact = self._get_holiday_effect(date_str)
        
        final_prob = base_rate + hour_impact + region_impact + holiday_impact + noise
        return int(np.random.random() < final_prob)
        
        final_prob = base_rate + hour_impact + region_impact + noise
        return int(np.random.random() < final_prob)
    
    def generate_campaign_data(self, start_date: str, days: int, calls_per_day: int) -> pd.DataFrame:
        """Génère un dataset complet"""
        records = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        for _ in range(days):
            for _ in range(calls_per_day):
                script = random.choice(list(self.scripts.keys()))
                region = random.choice(self.regions)
                product = random.choice(self.products)
                
                time_data = self._generate_call_time(current_date)
                
                record = {
                    'call_id': self.fake.uuid4(),
                    'region': region,
                    'product': product,
                    'script_version': script,
                    'duration': int(np.random.normal(300, 60)),
                    'agent_id': f"AG-{self.fake.random_int(1000, 9999)}",
                    **time_data,
                }
                
                record['converted'] = self._determine_conversion(
                    script, record['call_time'], region, record['call_date']
                )
                
                records.append(record)
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(records)
        
        # Validation
        self._validate_data(df)
        return df
    
    def _validate_data(self, df: pd.DataFrame):
        """Vérifie l'intégrité des données générées"""
        assert not df.duplicated('call_id').any(), "IDs dupliqués détectés"
        assert df['duration'].min() > 0, "Durées négatives invalides"
        assert set(df['script_version']) == set(self.scripts.keys()), "Scripts non reconnus"
        
        logger.info(f"Data validation passed. Generated {len(df)} records.")

    # Ajouter dans la classe :
    def add_holiday_effect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Simule l'impact des jours fériés"""
        holidays = ['2024-01-01', '2024-04-01', ...]
        df['is_holiday'] = df['call_date'].isin(holidays)
        df.loc[df['is_holiday'], 'converted'] *= 0.7  # Réduction de 30%
        return df
    
def parse_arguments():
        """Interface en ligne de commande professionnelle"""
        parser = argparse.ArgumentParser(description="Générateur de données pour campagnes commerciales")
        parser.add_argument('--start-date', type=str, required=True,
                        help='Date de début au format YYYY-MM-DD')
        parser.add_argument('--days', type=int, default=30,
                        help='Nombre de jours à générer')
        parser.add_argument('--calls-per-day', type=int, default=200,
                        help='Nombre d\'appels par jour')
        parser.add_argument('--output-dir', type=str, default='data/generated',
                        help='Répertoire de sortie')
        parser.add_argument('--seed', type=int, default=42,
                        help='Seed aléatoire pour la reproductibilité')
        return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_arguments()
    
    # Configuration du générateur
    generator = DataGenerator(seed=args.seed)
    
    # Génération des données
    df = generator.generate_campaign_data(
        start_date=args.start_date,
        days=args.days,
        calls_per_day=args.calls_per_day
    )
    
    # Création du répertoire de sortie
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export en CSV avec timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"campaign_data_{timestamp}.csv"
    df.to_csv(output_path, index=False)
    
    logger.info(f"Données générées avec succès dans {output_path}")