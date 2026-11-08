import os
from pdf2docx import Converter

BASE_PDF = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC.pdf'
LOOP_DIR = 'conversion_attempts'
os.makedirs(LOOP_DIR, exist_ok=True)

strategies = [
    {'name': 'default', 'kwargs': {}},
    {'name': 'connected_border_100', 'kwargs': {'connected_border_penalty': 100}},
    {'name': 'line_overlap_10', 'kwargs': {'line_overlap': 10.0}}
]

for strat in strategies:
    print(f"\n--- Generating DOCX with strategy: {strat['name']} ---")
    out_docx = os.path.join(LOOP_DIR, f"{strat['name']}.docx")
    
    try:
        cv = Converter(BASE_PDF)
        cv.convert(out_docx, **strat['kwargs'])
        cv.close()
        print(f"Saved: {out_docx}")
    except Exception as e:
        print(f"Strategy {strat['name']} failed: {e}")

print("\nAll automated conversion attempts completed.")
