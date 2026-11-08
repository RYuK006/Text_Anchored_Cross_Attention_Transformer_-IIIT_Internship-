import pypandoc
import os
from docx import Document
from docx.shared import Inches
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_column_layout(section, num_cols):
    sectPr = section._sectPr
    cols = sectPr.xpath('./w:cols')
    if not cols:
        cols = OxmlElement('w:cols')
        sectPr.append(cols)
    else:
        cols = cols[0]
    cols.set(qn('w:num'), str(num_cols))
    cols.set(qn('w:space'), '360')

print("Running Pandoc to parse LaTeX and generate native DOCX...")
# Use pandoc to convert LaTeX to Docx
# This will natively translate math equations to Word equations, handle citations, and formatting.
output_file = 'Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Pandoc.docx'
pypandoc.convert_file('main.tex', 'docx', outputfile=output_file, extra_args=['--mathml'])

print("Pandoc finished. Now applying IEEE 2-column layout and margins...")
doc = Document(output_file)

# Set Margins and 2-column layout for the whole document
# (The user will have to manually span the title/abstract across columns, but the rest will be formatted perfectly)
for section in doc.sections:
    section.top_margin = Inches(0.75)
    section.bottom_margin = Inches(1.0)
    section.left_margin = Inches(0.62)
    section.right_margin = Inches(0.62)
    set_column_layout(section, 2)

doc.save(output_file)
print(f"Success! Saved to {output_file}")
