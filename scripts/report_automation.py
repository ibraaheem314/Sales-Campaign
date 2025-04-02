import pandas as pd
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, Image
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import matplotlib.pyplot as plt
import seaborn as sns

def generate_daily_report():
    # Chargement des données
    df = pd.read_csv('../data/generated/campaign_data.csv')
    
    # Analyse des KPIs
    kpis = {
        'Conversion Rate': df['converted'].mean(),
        'Avg Duration': df['duration'].mean(),
        'Peak Hour': df.groupby('call_time')['converted'].mean().idxmax()
    }
    
    # Visualisations
    plt.figure(figsize=(10, 6))
    sns.barplot(
        x='call_time', 
        y='converted', 
        hue='script_version', 
        data=df,
        ci=None
    )
    plt.title('Taux de Conversion par Heure et Script')
    plt.savefig('conversion_by_hour.png', bbox_inches='tight')
    
    # Génération PDF
    doc = SimpleDocTemplate(
        "daily_report.pdf",
        pagesize=letter,
        title="Rapport Quotidien Campagne"
    )
    
    styles = getSampleStyleSheet()
    elements = []
    
    # Titre
    elements.append(Paragraph(
        f"Rapport Journalier - {datetime.today().strftime('%d/%m/%Y')}",
        styles['Title']
    ))
    
    # KPIs
    elements.append(Paragraph("Indicateurs Clés :", styles['Heading2']))
    kpi_table = Table(
        [[k, f"{v:.2f}" if isinstance(v, float) else v] for k, v in kpis.items()],
        colWidths=[200, 100]
    )
    elements.append(kpi_table)
    
    # Visualisation
    elements.append(Spacer(1, 12))
    elements.append(Paragraph("Performance par Heure :", styles['Heading2']))
    elements.append(Image('conversion_by_hour.png', width=400, height=300))
    
    doc.build(elements)

if __name__ == "__main__":
    generate_daily_report()