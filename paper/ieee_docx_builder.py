import re
import os
from docx import Document
from docx.shared import Pt, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.section import WD_SECTION
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
    cols.set(qn('w:space'), '360') # 0.25 inch column spacing (closer to IEEE)

def add_ieee_heading(doc, text, level):
    if level == 0:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(24)
        run.bold = False
    elif level == 1:
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(text.upper())
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
    elif level == 2:
        p = doc.add_paragraph()
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        run.italic = True
        
def add_ieee_paragraph(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10)

def main():
    doc = Document()
    
    # Configure initial section (1 column) for Title and Abstract
    sect = doc.sections[0]
    sect.top_margin = Inches(0.75)
    sect.bottom_margin = Inches(1.0)
    sect.left_margin = Inches(0.62)
    sect.right_margin = Inches(0.62)

    with open('main.tex', 'r', encoding='utf-8') as f:
        content = f.read()

    # Title
    title_match = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
    if title_match:
        add_ieee_heading(doc, title_match.group(1).replace('\n', ' ').strip(), 0)

    # Abstract
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    if abstract:
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.right_indent = Inches(0.5)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run = p.add_run("Abstract—" + abstract.group(1).replace('\n', ' ').strip())
        run.bold = True
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)

    # Add a continuous section break to switch to 2 columns
    new_sect = doc.add_section(WD_SECTION.CONTINUOUS)
    set_column_layout(new_sect, 2)
    
    # Re-apply margins to new section just in case
    new_sect.top_margin = Inches(0.75)
    new_sect.bottom_margin = Inches(1.0)
    new_sect.left_margin = Inches(0.62)
    new_sect.right_margin = Inches(0.62)

    sections = re.finditer(r'\\section\{(.*?)\}(.*?)(?=\\section\{|\\end\{document\}|$)', content, re.DOTALL)
    
    for match in sections:
        sec_title = match.group(1).strip()
        sec_content = match.group(2).strip()
        
        add_ieee_heading(doc, sec_title, 1)
        
        paragraphs = sec_content.split('\n\n')
        for p_text in paragraphs:
            img_match = re.search(r'\\includegraphics.*?\{(.*?)\}', p_text)
            if img_match:
                img_file = img_match.group(1)
                if os.path.exists(img_file):
                    p = doc.add_paragraph()
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r = p.add_run()
                    r.add_picture(img_file, width=Inches(3.3)) # Approx column width
            else:
                # Basic cleanup
                p_text = re.sub(r'\\cite\{.*?\}', '[Cite]', p_text)
                p_text = re.sub(r'\\ref\{.*?\}', '[Ref]', p_text)
                p_text = re.sub(r'\\textit\{(.*?)\}', r'\1', p_text)
                p_text = re.sub(r'\\textbf\{(.*?)\}', r'\1', p_text)
                p_text = re.sub(r'\$.*?\$', '[Eq]', p_text)
                p_text = re.sub(r'\\begin\{.*?\}|\\end\{.*?\}|\\label\{.*?\}|\\caption\{.*?\}', '', p_text)
                
                p_text = p_text.replace('\n', ' ').strip()
                if p_text and not p_text.startswith('\\'):
                    add_ieee_paragraph(doc, p_text)

    doc.save('Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Advanced.docx')
    print("Advanced generation complete.")

if __name__ == '__main__':
    main()
