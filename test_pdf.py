from pdf_reader import extract_text_from_pdf

text = extract_text_from_pdf("sample_resume.pdf")

print("Resume text extracted successfully!\n")
print(text[:3000])