from pathlib import Path
from PyPDF2 import PdfReader

pdf_path = Path('Harvested_Energy_Prediction_Technique_for_Solar-Powered_Wireless_Sensor_Networks.pdf')
out_path = Path('results') / 'proenergy_paper_text.txt'
out_path.parent.mkdir(parents=True, exist_ok=True)

reader = PdfReader(str(pdf_path))
with open(out_path, 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            f.write(f"\n\n--- PAGE {i+1} ---\n\n")
            f.write(text)

print(f"extracted text to {out_path}")
