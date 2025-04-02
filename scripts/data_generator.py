"""
Générateur de données complexes pour campagnes commerciales
Caractéristiques :
- Horaires précis avec heures, minutes, secondes
- Nombre d'appels/jour aléatoire (loi de Poisson)
- Saisonnalités multiples (hebdo, mensuelle, vacances)
- Performances des agents variables
- Profils clients réalistes
- Événements spéciaux (campagnes marketing)
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta, time
import random
import logging
from typing import Dict, Union, Any
import argparse
from pathlib import Path
from scipy.stats import poisson
from tqdm import tqdm

# Configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataGenerator:
    def __init__(self, seed=42):
        self.fake = Faker()
        np.random.seed(seed)
        random.seed(seed)
        
        # Configuration des motifs cachés
        self.patterns = {
            'golden_hours': [(14, 16, 0.35), (10, 11, 0.25)],
            'agent_tiers': {
                'novice': (0.15, 0.3),
                'intermediate': (0.25, 0.4),
                'expert': (0.35, 0.5)
            },
            'client_profiles': {
                'SME': (0.3, 0.05),
                'startup': (0.25, 0.1),
                'enterprise': (0.35, 0.03)
            }
        }
        
        # Configuration initiale
        self.products = ['SaaS', 'Hardware', 'Consulting', 'Support']
        self.regions = ['North', 'South', 'East', 'West']
        self.campaigns = {
            '2024': [
                ('2024-03-01', '2024-04-01', 0.15),
                ('2024-06-15', '2024-07-15', 0.2)
            ]
        }

    def _generate_call_time(self) -> time:
        """Génère un horaire réaliste entre 8h et 18h avec variations"""
        # 60% de chance dans les plages de pic
        if random.random() < 0.6:
            if random.choice([True, False]):
                hour = random.randint(10, 11)  # Matin
            else:
                hour = random.randint(14, 15)  # Après-midi
        else:
            hour = random.randint(8, 17)
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        # 5% d'appels hors horaires
        if random.random() < 0.05:
            hour = random.choice([7, 18, 19])
            minute = random.randint(0, 59)
        
        return time(
            hour=hour,
            minute=minute,
            second=second
        )

    def _generate_daily_calls(self, current_date: datetime) -> int:
        """Génère le nombre d'appels du jour avec variations réalistes"""
        base_calls = poisson.rvs(500)
        day_factor = 1.3 if current_date.weekday() in [0, 4] else 1
        return int(base_calls * day_factor)

    def _create_agent_pool(self, num_agents=50) -> Dict[str, Dict[str, Any]]:
        """Crée un pool d'agents avec des performances variables"""
        agents = {}
        tiers = list(self.patterns['agent_tiers'].keys())
        for _ in range(num_agents):
            tier = np.random.choice(tiers, p=[0.4, 0.4, 0.2])
            min_p, max_p = self.patterns['agent_tiers'][tier]
            agents[self.fake.unique.uuid4()] = {
                'tier': tier,
                'perf': np.random.uniform(min_p, max_p),
                'tenure': np.random.poisson(180)
            }
        return agents

    def _get_campaign_boost(self, date: datetime) -> float:
        """Calcule le boost des campagnes marketing"""
        boost = 0
        for campaign in self.campaigns.get(str(date.year), []):
            start = datetime.strptime(campaign[0], '%Y-%m-%d')
            end = datetime.strptime(campaign[1], '%Y-%m-%d')
            if start <= date <= end:
                boost += campaign[2]
        return boost

    def _determine_conversion(self, call_time: time, agent: Dict[str, Any], 
                             client_type: str, product: str, date: datetime) -> float:
        """Détermine la conversion avec des relations complexes"""
        # Extraction de l'heure et des minutes
        hour = call_time.hour
        minute = call_time.minute
        
        # Base rate
        base_rate = self.patterns['client_profiles'][client_type][0]
        
        # Facteur heure
        time_boost = next((b for s, e, b in self.patterns['golden_hours'] 
                          if s <= hour <= e), 0)
        
        # Pénalité pour les appels tardifs
        late_penalty = -0.15 if (hour == 17 and minute > 45) or hour > 17 else 0
        
        # Facteur agent
        agent_boost = agent['perf'] * (1 + agent['tenure']/365)
        
        # Facteur produit
        product_factor = 0.15 if product == 'Support' else 0.3
        
        # Facteur saisonnier
        day_of_year = date.timetuple().tm_yday
        seasonality = 0.1 * np.sin(2 * np.pi * day_of_year/365)
        
        # Combinaison non linéaire
        final_rate = (
            (base_rate * (1 + time_boost + agent_boost) + 
            product_factor + 
            seasonality + 
            self._get_campaign_boost(date) +
            late_penalty)
        )
        
        # Bruit aléatoire
        volatility = self.patterns['client_profiles'][client_type][1]
        final_rate += np.random.normal(0, volatility)
        
        return np.clip(final_rate, 0, 1)

    def generate_dataset(self, start_date: str, days: int) -> pd.DataFrame:
        """Génère le dataset complet avec horaires précis"""
        agents = self._create_agent_pool()
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        records = []
        
        for _ in tqdm(range(days), desc="Génération des données"):
            daily_calls = self._generate_daily_calls(current_date)
            client_types = np.random.choice(
                list(self.patterns['client_profiles'].keys()), 
                daily_calls,
                p=[0.5, 0.3, 0.2]
            )
            
            for _ in range(daily_calls):
                agent_id = random.choice(list(agents.keys()))
                product = np.random.choice(self.products, p=[0.4, 0.3, 0.2, 0.1])
                call_time = self._generate_call_time()
                
                conversion_prob = self._determine_conversion(
                    call_time=call_time,
                    agent=agents[agent_id],
                    client_type=np.random.choice(client_types),
                    product=product,
                    date=current_date
                )
                
                records.append({
                    'call_id': self.fake.uuid4(),
                    'call_datetime': datetime.combine(current_date, call_time).isoformat(),
                    'call_time': call_time.strftime("%H:%M:%S"),
                    'call_hour': call_time.hour,
                    'duration': int(np.random.gamma(3, 100)),
                    'product': product,
                    'region': np.random.choice(self.regions),
                    'agent_id': agent_id,
                    'agent_tier': agents[agent_id]['tier'],
                    'client_type': np.random.choice(client_types),
                    'previous_contacts': np.random.poisson(1.2),
                    'converted': int(np.random.random() < conversion_prob),
                    'campaign_boost': self._get_campaign_boost(current_date)
                })
            
            current_date += timedelta(days=1)
        
        df = pd.DataFrame(records)
        return self._add_realistic_noise(df)

    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des artefacts réalistes complexes"""
    
        # 1. Valeurs manquantes aléatoires
        missing_patterns = {
        'duration': 0.03,  # 3% de durées manquantes
        'product': 0.01,   # 1% de produits manquants
        'region': 0.02,    # 2% de régions manquantes
        'agent_id': 0.005  # 0.5% d'agents non enregistrés
        }
    
        for col, frac in missing_patterns.items():
            missing_idx = df.sample(frac=frac).index
            df.loc[missing_idx, col] = None
    
        # 2. Durées aberrantes avec différents patterns
        duration_anomalies = df.sample(frac=0.02).index
        df.loc[duration_anomalies, 'duration'] = np.where(
            np.random.rand(len(duration_anomalies)) > 0.5,
            df.loc[duration_anomalies, 'duration'] * 10,  # Valeurs extrêmes
            0  # Durées nulles
            )
    
        # 3. Incohérences temporelles complexes
        time_issues = df.sample(frac=0.01).index
        df.loc[time_issues, 'call_time'] = np.random.choice([
            "00:00:00", 
            "23:59:59",
            "12:00:00",
            "99:99:99"], size=len(time_issues))
    
        # 4. Incohérences de produits/régions
        product_region_mismatch = df.sample(frac=0.005).index
        df.loc[product_region_mismatch, 'product'] = np.where(
            df.loc[product_region_mismatch, 'region'] == 'North',
            'Hardware',  # Forcer un produit incohérent
            'SaaS')
    
        # 5. Doublons partiels
        duplicate_idx = df.sample(frac=0.003).index
        df = pd.concat([
            df,
            df.loc[duplicate_idx].assign(call_id=lambda x: x['call_id'] + '_dup')])
    
        # 6. Valeurs de conversion incohérentes
        conversion_issues = df.sample(frac=0.01).index
        df.loc[conversion_issues, 'converted'] = np.where(
            (df.loc[conversion_issues, 'duration'] < 30) | 
            (df.loc[conversion_issues, 'duration'] > 1800),
            1,  # Conversion improbable pour durée extrême
            df.loc[conversion_issues, 'converted'] )
    
        # 7. Problèmes de formats
        format_issues = df.sample(frac=0.005).index
        df.loc[format_issues, 'agent_id'] = df.loc[format_issues, 'agent_id'].str.replace('AG-', 'Agent')
    
        # 8. Anomalies saisonnières (plus d'erreurs en décembre)
        if not df.empty:
            december_idx = df[df['call_datetime'].str.contains('-12-')].sample(frac=0.05).index
            df.loc[december_idx, 'duration'] = df.loc[december_idx, 'duration'] * 5
    
        return df

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', required=True)
    parser.add_argument('--days', type=int, default=365)
    parser.add_argument('--output', default='data/campaign_data.csv')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generator = DataGenerator()
    
    logger.info("Génération du dataset...")
    df = generator.generate_dataset(args.start_date, args.days)
    
    logger.info("\n Statistiques globales :")
    logger.info(f"Taux de conversion : {df['converted'].mean():.2%}")
    logger.info(f"Exemple d'horaire : {df['call_time'].iloc[0]}")
    logger.info(f"Répartition horaire :\n{df['call_hour'].value_counts().sort_index()}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Dataset sauvegardé dans {output_path}")