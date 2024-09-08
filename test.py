import openai
import os
import json
from PyPDF2 import PdfReader

# Set up OpenAI API key
openai.api_key = 'your-openai-api-key'

def extract_text_from_pdf(pdf_path):
    """
    Function to extract text from a PDF file
    """
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def extract_information_from_paper(text):
    """
    Function to send the extracted PDF text to OpenAI and get the structured JSON response
    """
    prompt = f"""(!) AFTER CAREFULLY READING AND ANALYZING THIS RESEARCH PAPER (!), 
    USING ALL THE INFORMATION GIVEN IN THIS RESEARCH PAPER (!), 
    BASED ON ALL THE INFORMATION GIVEN IN THIS RESEARCH PAPER (!), 
    PLEASE, EXTRACT THE FOLLOWING INFORMATION IN THE FOLLOWING JSON FORMAT:
    {{
      "title": String, // Paper Title
      "subtitle": String, // Paper Subtitle if exists; otherwise, don't include this key
      "authors": Array([{
        "name": String, // Full Name of the Author
        "email": String, // Email Address of the Author if exist; otherwise, don't include this key
        "institution": String, // Author's Affiliated Institution
        "department": String, // That institution's department the Author belongs to, if exist; otherwise, don't include this key
      }, ...]), 
      "summary": String, // A paragraph-long summary of the paper (combine and summarize abstract and introduction)
      "abstract": String, // The whole full Abstract of the Paper
      "introduction": String, // The whole full Introduction of the Paper
      "links": Array([String]), // All the URLs mentioned in the Paper
      "references": Array([String]), // All the references listed at the end of the Paper
      "citation": String // How this Paper would be used as a Reference similar to the ones inside the references - generate similarly-formatted citation, reference string for this Paper
    }}"""

    # Combine the text with the prompt
    prompt_with_text = f"{prompt}\n\nText of the research paper: {text}"

    response = openai.Completion.create(
        engine="gpt-4",  # Replace with the engine of your choice, e.g., "gpt-3.5-turbo"
        prompt=prompt_with_text,
        max_tokens=4096,  # Set maximum tokens accordingly
        temperature=0.2,
    )

    # Parse the JSON response
    return json.loads(response.choices[0].text)

def process_multiple_pdfs(pdf_paths):
    """
    Process multiple PDFs and return a list of JSON objects containing the extracted information
    """
    extracted_data = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path}...")
        # Step 1: Extract text from the PDF
        text = extract_text_from_pdf(pdf_path)
        
        # Step 2: Extract structured information using OpenAI API
        paper_info = extract_information_from_paper(text)
        extracted_data.append(paper_info)
    
    return extracted_data

# Example usage
if __name__ == "__main__":
    # List of PDF file paths
    pdf_files = ["paper1.pdf", "paper2.pdf", "paper3.pdf"]

    # Process the PDFs and get the extracted information
    json_results = process_multiple_pdfs(pdf_files)

    # Print or save the JSON results
    print(json.dumps(json_results, indent=4))
    
    # Optionally, save results to a file
    with open("extracted_papers_info.json", "w") as outfile:
        json.dump(json_results, outfile, indent=4)
