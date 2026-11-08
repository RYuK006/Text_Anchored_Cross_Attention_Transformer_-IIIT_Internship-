import os
import fitz
import cv2
import numpy as np
from docx import Document
from docx.shared import Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from spire.doc import Document as SpireDocument, FileFormat
from skimage.metrics import structural_similarity as ssim
import shutil

BASE_DOCX = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Pandoc.docx'
BASE_PDF = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC.pdf'
LOOP_DIR = 'visual_optimization'
BASELINE_IMG_DIR = os.path.join(LOOP_DIR, 'baseline')

os.makedirs(BASELINE_IMG_DIR, exist_ok=True)

def render_pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for i in range(min(2, len(doc))): # first 2 pages is enough for layout test
        page = doc[i]
        pix = page.get_pixmap(dpi=100) # lower dpi for faster ssim
        out_path = os.path.join(output_dir, f'page_{i}.png')
        pix.save(out_path)
        image_paths.append(out_path)
    return image_paths

def set_column_layout(section, num_cols, spacing):
    sectPr = section._sectPr
    cols = sectPr.xpath('./w:cols')
    if not cols:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    else:
        cols = cols[0]
    cols.set(qn('w:num'), str(num_cols))
    cols.set(qn('w:space'), str(spacing))

def calculate_similarity(baseline_paths, test_paths):
    scores = []
    for bp, tp in zip(baseline_paths, test_paths):
        img_b = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
        img_t = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
        if img_b.shape != img_t.shape:
            img_t = cv2.resize(img_t, (img_b.shape[1], img_b.shape[0]))
        score, _ = ssim(img_b, img_t, full=True)
        scores.append(score)
    return np.mean(scores)

print("Extracting baseline PDF images...")
baseline_imgs = render_pdf_to_images(BASE_PDF, BASELINE_IMG_DIR)

param_grid = [
    {'name': 'margins_ieee_space_narrow', 'top': 0.75, 'bot': 1.0, 'side': 0.62, 'space': 200},
    {'name': 'margins_ieee_space_standard', 'top': 0.75, 'bot': 1.0, 'side': 0.62, 'space': 360},
    {'name': 'margins_wide_space_standard', 'top': 1.0, 'bot': 1.0, 'side': 1.0, 'space': 360},
    {'name': 'margins_ieee_space_wide', 'top': 0.75, 'bot': 1.0, 'side': 0.62, 'space': 720},
]

best_score = -1.0
best_name = ""

for param in param_grid:
    print(f"\n--- Testing iteration: {param['name']} ---")
    doc = Document(BASE_DOCX)
    
    for section in doc.sections:
        section.top_margin = Inches(param['top'])
        section.bottom_margin = Inches(param['bot'])
        section.left_margin = Inches(param['side'])
        section.right_margin = Inches(param['side'])
        set_column_layout(section, 2, param['space'])
        
    temp_docx = os.path.join(LOOP_DIR, f"{param['name']}.docx")
    doc.save(temp_docx)
    
    # Render with Spire.Doc
    temp_pdf = os.path.join(LOOP_DIR, f"{param['name']}.pdf")
    try:
        sd = SpireDocument()
        sd.LoadFromFile(temp_docx)
        sd.SaveToFile(temp_pdf, FileFormat.PDF)
        
        # Render to images
        img_dir = os.path.join(LOOP_DIR, f"{param['name']}_imgs")
        test_imgs = render_pdf_to_images(temp_pdf, img_dir)
        
        score = calculate_similarity(baseline_imgs, test_imgs)
        print(f"Visual Match Score (SSIM): {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_name = param['name']
            
    except Exception as e:
        print(f"Failed to evaluate {param['name']}: {e}")

print(f"\n=================================")
print(f"Optimal parameters found: {best_name} with SSIM = {best_score:.4f}")
best_docx_path = os.path.join(LOOP_DIR, f"{best_name}.docx")
final_dest = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Final_Match.docx'
if os.path.exists(best_docx_path):
    shutil.copy(best_docx_path, final_dest)
    print(f"Saved the highest fidelity DOCX to {final_dest}")
