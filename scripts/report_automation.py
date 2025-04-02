from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image
from reportlab.lib.styles import getSampleStyleSheet
import pandas as pd

def create_report():
    df = pd.read_csv('data/sample_data/call_logs.csv')
    
    # Calculs
    conversion_rate = df['converted'].mean() * 100
    peak_hour = df.groupby('call_time')['converted'].mean().idxmax()
    
    # Génération PDF
    doc = SimpleDocTemplate("daily_report.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    
    content = []
    content.append(Paragraph(f"Conversion Rate: {conversion_rate:.1f}%", styles['Title']))
    content.append(Paragraph(f"Optimal Call Hour: {peak_hour}h", styles['BodyText']))
    
    doc.build(content)

if __name__ == "__main__":
    create_report()