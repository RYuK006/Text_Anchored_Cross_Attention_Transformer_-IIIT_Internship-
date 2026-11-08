import re
from docx import Document
from docx.shared import Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

def set_number_of_columns(section, num_cols):
    sectPr = section._sectPr
    cols = sectPr.xpath('./w:cols')[0]
    cols.set(qn('w:num'), str(num_cols))
    cols.set(qn('w:space'), '708') # standard 0.5 inch spacing

def parse_latex(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    title = re.search(r'\\title\{(.*?)\}', content, re.DOTALL)
    title = title.group(1).strip() if title else "Title"
    
    # Strip some common latex commands from text
    content = re.sub(r'\\cite\{.*?\}', '[Cite]', content)
    content = re.sub(r'\\ref\{.*?\}', '[Ref]', content)
    content = re.sub(r'\\textit\{(.*?)\}', r'\1', content)
    content = re.sub(r'\\textbf\{(.*?)\}', r'\1', content)
    
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', content, re.DOTALL)
    abstract = abstract.group(1).strip() if abstract else ""
    
    sections = []
    for match in re.finditer(r'\\section\{(.*?)\}(.*?)(?=\\section\{|\\end\{document\}|$)', content, re.DOTALL):
        sections.append((match.group(1).strip(), match.group(2).strip()))
        
    return title, abstract, sections

doc = Document()
section = doc.sections[0]
set_number_of_columns(section, 2)

title, abstract, sections = parse_latex('main.tex')

title_para = doc.add_heading(title.replace('\n', ' '), 0)

if abstract:
    doc.add_heading('Abstract', level=1)
    doc.add_paragraph(abstract.replace('\n', ' '))

for sec_title, sec_content in sections:
    doc.add_heading(sec_title, level=1)
    subsecs = re.split(r'\\subsection\{(.*?)\}', sec_content)
    
    if subsecs[0].strip():
        for p in subsecs[0].split('\n\n'):
            p = p.strip()
            if p and not p.startswith('\\begin') and not p.startswith('\\end'):
                doc.add_paragraph(p.replace('\n', ' '))
                
    for i in range(1, len(subsecs), 2):
        doc.add_heading(subsecs[i], level=2)
        for p in subsecs[i+1].split('\n\n'):
            p = p.strip()
            if p and not p.startswith('\\begin') and not p.startswith('\\end'):
                doc.add_paragraph(p.replace('\n', ' '))

doc.save('Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Scripted.docx')
print("Conversion complete!")
