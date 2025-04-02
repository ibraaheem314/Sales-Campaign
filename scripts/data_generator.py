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
from typing import Dict, Any
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
            },
            'city_effects': {
                'San Francisco': 0.1,
                'Detroit': -0.05,
                'New York': 0.15,
                'Houston': 0.07
            }
        }
        
        # Configuration géographique
        self.usa_cities = {
            'West': ['Los Angeles', 'San Francisco', 'Seattle', 'San Diego', 'Las Vegas'],
            'South': ['Houston', 'Dallas', 'Miami', 'Atlanta', 'New Orleans'],
            'East': ['New York', 'Boston', 'Philadelphia', 'Washington', 'Baltimore'],
            'North': ['Chicago', 'Detroit', 'Minneapolis', 'Milwaukee', 'Indianapolis']
        }
        self.regions = list(self.usa_cities.keys())
        
        # Autres configurations
        self.products = ['SaaS', 'Hardware', 'Consulting', 'Support']
        self.campaigns = {
            '2024': [
                ('2024-03-01', '2024-04-01', 0.15),
                ('2024-06-15', '2024-07-15', 0.2)
            ]
        }

    def _get_random_city(self, region: str) -> str:
        """Génère une ville avec une distribution réaliste mais plus aléatoire"""
        cities = self.usa_cities[region]
    
            # 1. Poids de base avec variation aléatoire
        base_weights = {
        'New York': 17.584, 'Los Angeles': 13.02871, 'Chicago': 11.5648,  # Grandes métropoles
        'San Francisco': 8.1202, 'Houston': 7.698, 'Miami': 4.39814,      # Villes importantes
        'default': 1.254                                      # Autres villes
            }
    
        # 2. Application des poids avec variation aléatoire (±30%)
        weights = []
        for city in cities:
            if city in base_weights:
                base_weight = base_weights[city]
                # Variation aléatoire entre 70% et 130% du poids de base
                variation = 0.7 + 0.6 * random.random()  # Entre 0.7 et 1.3
                weight = base_weight * variation
            else:
                # Variation plus forte pour les petites villes
                variation = 0.5 + random.random()  # Entre 0.5 et 1.5
                weight = base_weights['default'] * variation
        
            weights.append(weight)
    
        # 3. 10% de chance de sélection totalement aléatoire
        if random.random() < 0.1:
            return random.choice(cities)
    
        # 4. 5% de chance de retourner une ville d'une autre région
        if random.random() < 0.05:
            other_regions = [r for r in self.regions if r != region]
            foreign_region = random.choice(other_regions)
            return random.choice(self.usa_cities[foreign_region])
    
        return random.choices(cities, weights=weights, k=1)[0]

    def _generate_call_time(self) -> time:
        """Génère un horaire réaliste entre 8h et 18h avec variations"""
        if random.random() < 0.6:
            hour = random.choice([10, 11] if random.random() < 0.5 else [14, 15])
        else:
            hour = random.randint(8, 17)
        
        # 5% d'appels hors horaires
        if random.random() < 0.05:
            hour = random.choice([7, 18, 19])
        
        return time(
            hour=hour,
            minute=random.randint(0, 59),
            second=random.randint(0, 59)
        )

    def _generate_daily_calls(self, current_date: datetime) -> int:
        """Génère le nombre d'appels du jour avec variations réalistes"""
        return int(poisson.rvs(500) * (1.3 if current_date.weekday() in [0, 4] else 1))

    def _create_agent_pool(self, num_agents=50) -> Dict[str, Dict[str, Any]]:
        """Crée un pool d'agents avec des performances variables"""
        agents = {}
        tiers = list(self.patterns['agent_tiers'].keys())
        for _ in range(num_agents):
            tier = np.random.choice(tiers, p=[0.132, 0.388, 0.48])
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

    def _determine_conversion(self, **kwargs) -> float:
        """Détermine la conversion avec des relations complexes"""
        # Extraction des paramètres
        call_time = kwargs.get('call_time', time(12, 0))
        agent = kwargs.get('agent', {'perf': 0.3, 'tenure': 180})
        client_type = kwargs.get('client_type', 'SME')
        product = kwargs.get('product', 'SaaS')
        date = kwargs.get('date', datetime.now())
        city = kwargs.get('city', 'Unknown')

        # Calcul des effets
        hour = call_time.hour
        base_rate = self.patterns['client_profiles'][client_type][0]
        time_boost = next((b for s, e, b in self.patterns['golden_hours'] if s <= hour <= e), 0)
        city_boost = self.patterns['city_effects'].get(city, 0)
        late_penalty = -0.15 if (hour == 17 and call_time.minute > 45) or hour > 17 else 0
        agent_boost = agent['perf'] * (1 + agent['tenure']/365)
        product_factor = 0.15 if product == 'Support' else 0.3
        seasonality = 0.1 * np.sin(2 * np.pi * date.timetuple().tm_yday/365)

        # Calcul final
        final_rate = (base_rate * (1 + time_boost + agent_boost) + product_factor + \
                    seasonality + self._get_campaign_boost(date) + late_penalty + city_boost)
        final_rate += np.random.normal(0, self.patterns['client_profiles'][client_type][1])
        
        return np.clip(final_rate, 0, 1)

    def generate_dataset(self, start_date: str, days: int) -> pd.DataFrame:
        """Génère le dataset complet avec horaires précis"""
        agents = self._create_agent_pool()
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        records = []
        
        for _ in tqdm(range(days), desc="Génération des données"):
            daily_calls = self._generate_daily_calls(current_date)
            
            for _ in range(daily_calls):
                region = np.random.choice(self.regions)
                city = self._get_random_city(region)
                agent_id = random.choice(list(agents.keys()))
                product = np.random.choice(self.products, p=[0.245, 0.445, 0.136, 0.174])
                call_time = self._generate_call_time()
                client_type = np.random.choice(list(self.patterns['client_profiles'].keys()), 
                                              p=[0.315, 0.535, 0.15])

                conversion_prob = self._determine_conversion(
                    call_time=call_time,
                    agent=agents[agent_id],
                    client_type=client_type,
                    product=product,
                    date=current_date,
                    city=city
                )
                
                records.append({
                    'call_id': self.fake.uuid4(),
                    'call_datetime': datetime.combine(current_date, call_time).isoformat(),
                    'call_time': call_time.strftime("%H:%M:%S"),
                    'call_hour': call_time.hour,
                    'duration': int(np.random.gamma(3, 100)),
                    'product': product,
                    'region': region,
                    'city': city,
                    'agent_id': agent_id,
                    'agent_tier': agents[agent_id]['tier'],
                    'client_type': client_type,
                    'previous_contacts': np.random.poisson(1.2),
                    'converted': int(np.random.random() < conversion_prob),
                    'campaign_boost': self._get_campaign_boost(current_date)
                })
            
            current_date += timedelta(days=1)
        
        return self._add_realistic_noise(pd.DataFrame(records))

    def _add_realistic_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ajoute des artefacts réalistes complexes de manière robuste"""
        try:
            # Créer une copie pour éviter les SettingWithCopyWarning
            df = df.copy()
        
            # 1. Valeurs manquantes différentielles
            missing_config = {
                'duration': 0.15, 'product': 0.1, 'region': 0.05,
                'agent_id': 0.02, 'client_type': 0.07
                    }
        
            for col, rate in missing_config.items():
                if col in df.columns:
                    mask = np.random.random(len(df)) < rate
                    df.loc[mask, col] = None

            # 2. Anomalies temporelles (plus robuste)
            if 'call_time' in df.columns:
                time_mask = np.random.random(len(df)) < 0.01
                df.loc[time_mask, 'call_time'] = np.random.choice(
                    ["00:00:00", "23:59:59", "12:00:00", "99:99:99"],
                    size=time_mask.sum()
                    )

            # 3. Incohérences métier (version corrigée)
            if all(col in df.columns for col in ['product', 'region']):
                mismatch_mask = np.random.random(len(df)) < 0.005
                df.loc[mismatch_mask, 'product'] = np.where(
                    df.loc[mismatch_mask, 'region'] == 'North',
                    'Hardware',
                    'SaaS'
                        )

            # 4. Données extrêmes
            if 'duration' in df.columns:
                extreme_mask = np.random.random(len(df)) < 0.02
                df.loc[extreme_mask, 'duration'] = df.loc[extreme_mask, 'duration'] * 10

            # 5. Problèmes de formats (exemple corrigé)
            if 'agent_id' in df.columns:
                format_mask = np.random.random(len(df)) < 0.01
                df.loc[format_mask, 'agent_id'] = 'AG-' + df.loc[format_mask, 'agent_id'].astype(str)

            return df

        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de bruit : {str(e)}")
            return df  # Retourne les données non modifiées en cas d'erreur

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start-date', required=True, help='Date de début au format YYYY-MM-DD')
    parser.add_argument('--days', type=int, default=365, help='Nombre de jours à générer')
    parser.add_argument('--output', default='data/campaign_data.csv', help='Chemin de sortie')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    generator = DataGenerator()
    
    logger.info("Début de la génération des données...")
    df = generator.generate_dataset(args.start_date, args.days)
    
    logger.info("\nAnalyse initiale :")
    logger.info(f"Taux de conversion moyen : {df['converted'].mean():.2%}")
    logger.info(f"Villes les plus fréquentes :\n{df['city'].value_counts().head(5)}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Données sauvegardées dans {output_path}")