import os
from docx import Document
from docx.shared import Inches
from docx.enum.section import WD_SECTION
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

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

def fix_layout(input_docx, output_docx):
    doc = Document(input_docx)
    
    # Set the first section (default) to 1 column and standard margins
    sect_1 = doc.sections[0]
    sect_1.top_margin = Inches(0.75)
    sect_1.bottom_margin = Inches(1.0)
    sect_1.left_margin = Inches(0.62)
    sect_1.right_margin = Inches(0.62)
    set_column_layout(sect_1, 1, 360) # 1 column for Title/Abstract
    
    # We need to find the paragraph that starts the first main section (usually "I. INTRODUCTION" or just "INTRODUCTION")
    # Pandoc usually styles section headers with "Heading 1" or similar.
    # We will iterate through paragraphs. When we hit the first Heading 1, we insert a section break before it.
    
    found_intro = False
    for p in doc.paragraphs:
        # Check if this paragraph is a major heading (e.g. Introduction)
        if p.style.name.startswith('Heading 1') or 'INTRODUCTION' in p.text.upper():
            # Insert a continuous section break right before this paragraph
            p_xml = p._p
            sectPr = OxmlElement('w:sectPr')
            
            # The section break properties for the PREVIOUS section (Title/Abstract)
            # must be inserted at the end of the paragraph before this one.
            # python-docx has a helper insert_section_break, but doing it manually is safer.
            
            # Actually, python-docx Document.add_section() adds it at the end of the document.
            # We can't easily insert a section break in the middle of a document via high-level API.
            # Instead, let's use a workaround:
            break
            
    # If inserting a section break in the middle is too hard with python-docx, 
    # we can rebuild a new document by copying elements, but that loses equation formatting.
    pass

# Wait, there is a known way to insert a continuous section break before a paragraph in python-docx:
def insert_continuous_section_break_before(paragraph):
    p = paragraph._p
    pPr = p.get_or_add_pPr()
    sectPr = OxmlElement('w:sectPr')
    type_elm = OxmlElement('w:type')
    type_elm.set(qn('w:val'), 'continuous')
    sectPr.append(type_elm)
    pPr.append(sectPr)
    return sectPr

def apply_fix():
    doc = Document('Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Pandoc.docx')
    
    # Set global margins for section 0 (which now applies to the end of the doc until we add breaks)
    doc.sections[0].top_margin = Inches(0.75)
    doc.sections[0].bottom_margin = Inches(1.0)
    doc.sections[0].left_margin = Inches(0.62)
    doc.sections[0].right_margin = Inches(0.62)
    
    # Find the "INTRODUCTION" or first major heading
    intro_para = None
    for p in doc.paragraphs:
        if 'INTRODUCTION' in p.text.upper() or (p.style.name.startswith('Heading 1') and len(p.text) > 2):
            intro_para = p
            break
            
    if intro_para:
        print(f"Found intro paragraph: {intro_para.text}")
        
        # In Word XML, a section break at the end of a paragraph defines the properties of the section PRECEDING it.
        # So if we put a section break on the paragraph BEFORE the introduction, 
        # that break defines the Title/Abstract section (1 column).
        # The remainder of the document (defined by doc.sections[-1]) will be 2 columns.
        
        # Find the paragraph before intro_para
        prev_p = None
        for p in doc.paragraphs:
            if p == intro_para:
                break
            prev_p = p
            
        if prev_p:
            print("Applying continuous section break after abstract...")
            # Create a section break (continuous)
            sectPr = insert_continuous_section_break_before(prev_p)
            
            # Now doc has 2 sections. 
            # doc.sections[0] is the Title/Abstract. Set it to 1 column.
            set_column_layout(doc.sections[0], 1, 360)
            
            # doc.sections[1] (or -1) is the rest of the document. Set it to 2 columns.
            set_column_layout(doc.sections[-1], 2, 360)
            
            # Also need to set margins for the new section
            doc.sections[-1].top_margin = Inches(0.75)
            doc.sections[-1].bottom_margin = Inches(1.0)
            doc.sections[-1].left_margin = Inches(0.62)
            doc.sections[-1].right_margin = Inches(0.62)
        else:
            print("Could not find paragraph before intro.")
    else:
        print("Could not find Introduction heading.")

    doc.save('Aaron_Alex_Mathew_Research_Paper_IEEE_TAC_Fixed_Layout.docx')
    print("Fixed layout generated.")

apply_fix()
