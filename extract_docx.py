import zipfile
import xml.etree.ElementTree as ET

def get_docx_text(path):
    text = []
    try:
        with zipfile.ZipFile(path) as docx:
            xml_content = docx.read('word/document.xml')
            tree = ET.XML(xml_content)
            
            WORD_NAMESPACE = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
            PARA = WORD_NAMESPACE + 'p'
            TEXT = WORD_NAMESPACE + 't'
            
            for paragraph in tree.iter(PARA):
                texts = [node.text for node in paragraph.iter(TEXT) if node.text]
                if texts:
                    text.append(''.join(texts))
    except Exception as e:
        print(f"Error reading {path}: {e}")
    return '\n'.join(text)

template = get_docx_text(r"C:\Users\Aaron\Desktop\IIIT-Kottayam\template\Transactions-template-and-instructions-on-how-to-create-your-article-formatted (4).docx")
report = get_docx_text(r"C:\Users\Aaron\Desktop\IIIT-Kottayam\Aaron Alex Mathew Report.docx")

with open(r"C:\Users\Aaron\Desktop\IIIT-Kottayam\template_text.txt", "w", encoding="utf-8") as f:
    f.write(template)

with open(r"C:\Users\Aaron\Desktop\IIIT-Kottayam\report_text.txt", "w", encoding="utf-8") as f:
    f.write(report)

print("Successfully extracted text from both docx files!")
