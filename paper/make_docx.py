import fitz
from docx import Document
from docx.shared import Pt
import os

pdf_path = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC.pdf'
docx_path = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_ImageBased.docx'

doc = fitz.open(pdf_path)
document = Document()

# Configure the first section
sect = document.sections[0]
p = doc[0]
sect.page_width = Pt(p.rect.width)
sect.page_height = Pt(p.rect.height)
sect.top_margin = Pt(0)
sect.bottom_margin = Pt(0)
sect.left_margin = Pt(0)
sect.right_margin = Pt(0)
sect.header_distance = Pt(0)
sect.footer_distance = Pt(0)

for i in range(len(doc)):
    page = doc[i]
    pix = page.get_pixmap(dpi=300)
    img_path = f'temp_page_{i}.png'
    pix.save(img_path)
    
    document.add_picture(img_path, width=Pt(page.rect.width))
    os.remove(img_path)

document.save(docx_path)
print('Done')
