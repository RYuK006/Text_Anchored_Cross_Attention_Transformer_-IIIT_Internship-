import os
import fitz
import numpy as np
import cv2
from pdf2docx import Converter
from docx2pdf import convert
import shutil
from skimage.metrics import structural_similarity as ssim

BASE_PDF = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC.pdf'
LOOP_DIR = 'conversion_loop'
BASELINE_IMG_DIR = os.path.join(LOOP_DIR, 'baseline')

os.makedirs(BASELINE_IMG_DIR, exist_ok=True)

def render_pdf_to_images(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    # Just check first 3 pages to save time, or all if we want
    for i in range(min(3, len(doc))):
        page = doc[i]
        pix = page.get_pixmap(dpi=150)
        out_path = os.path.join(output_dir, f'page_{i}.png')
        pix.save(out_path)
        image_paths.append(out_path)
    return image_paths

def calculate_similarity(baseline_paths, test_paths):
    scores = []
    for bp, tp in zip(baseline_paths, test_paths):
        img_b = cv2.imread(bp, cv2.IMREAD_GRAYSCALE)
        img_t = cv2.imread(tp, cv2.IMREAD_GRAYSCALE)
        
        # Resize img_t to match img_b if dimensions differ
        if img_b.shape != img_t.shape:
            img_t = cv2.resize(img_t, (img_b.shape[1], img_b.shape[0]))
            
        score, _ = ssim(img_b, img_t, full=True)
        scores.append(score)
    return np.mean(scores)

print("Rendering baseline PDF images...")
baseline_imgs = render_pdf_to_images(BASE_PDF, BASELINE_IMG_DIR)

strategies = [
    {'name': 'default', 'kwargs': {}},
    {'name': 'connected_border_100', 'kwargs': {'connected_border_penalty': 100}},
    {'name': 'line_overlap_10', 'kwargs': {'line_overlap': 10.0}}
]

best_score = -1.0
best_strategy = None
best_docx = None

for strat in strategies:
    print(f"\n--- Trying strategy: {strat['name']} ---")
    out_docx = os.path.join(LOOP_DIR, f"{strat['name']}.docx")
    out_pdf = os.path.join(LOOP_DIR, f"{strat['name']}.pdf")
    img_dir = os.path.join(LOOP_DIR, f"{strat['name']}_images")
    
    try:
        cv = Converter(BASE_PDF)
        cv.convert(out_docx, **strat['kwargs'])
        cv.close()
        
        print("Converting DOCX back to PDF for visual analysis...")
        convert(out_docx, out_pdf)
        
        print("Rendering DOCX-PDF to images...")
        test_imgs = render_pdf_to_images(out_pdf, img_dir)
        
        score = calculate_similarity(baseline_imgs, test_imgs)
        print(f"SSIM Score: {score:.4f} (Higher is closer to exact copy, 1.0 is perfect)")
        
        if score > best_score:
            best_score = score
            best_strategy = strat['name']
            best_docx = out_docx
            
    except Exception as e:
        print(f"Strategy {strat['name']} failed: {e}")

print(f"\n=============================")
print(f"Best Strategy Found: {best_strategy} with SSIM Score: {best_score:.4f}")
print("Copying best DOCX to root directory...")
if best_docx:
    shutil.copy(best_docx, 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_BestMatch.docx')
print("Loop completed.")
