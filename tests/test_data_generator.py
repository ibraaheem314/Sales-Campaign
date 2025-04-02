import pytest
from data_generator import DataGenerator
import pandas as pd

class TestDataGenerator:
    @pytest.fixture
    def generator(self):
        return DataGenerator(seed=42)

    def test_holiday_effect(self, generator):
        # Test jour férié
        assert generator._get_holiday_effect('2024-07-14') == -0.4
        # Test jour normal
        assert generator._get_holiday_effect('2024-01-02') == 0.0

    def test_data_integrity(self, generator):
        df = generator.generate_campaign_data('2024-01-01', 2, 100)
        assert len(df) == 200
        assert df['converted'].isin([0, 1]).all()
        
    def test_region_targeting(self, generator):
        df = generator.generate_campaign_data('2024-01-01', 1, 500)
        script_a_regions = df[df['script_version'] == 'A']['region'].unique()
        assert set(script_a_regions).issubset(['North', 'East'])

    def test_weekend_effect(self, generator):
        # Génération un samedi
        df = generator.generate_campaign_data('2024-01-06', 1, 200)
        weekend_hours = df['call_time'].unique()
        assert all(h >= 10 for h in weekend_hours)