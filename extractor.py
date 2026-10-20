import os
import json
from google import genai

client = genai.Client()

def extract_with_gemini(document_text: str):
    prompt = f"""
    You are an expert document information extraction system. 
    Analyze the following unstructured document text and extract key fields into a clean JSON schema.
    
    Extract these fields if present:
    - document_type (e.g., Invoice, Receipt, Contract)
    - sender_company
    - recipient_company
    - document_number / invoice_number
    - dates (list any dates found)
    - total_amount
    - key_items_or_summary

    Return ONLY valid JSON with no extra markdown formatting or conversational text.

    Document Text:
    """ + document_text

    # We use gemini-3.8-flash, with a fallback block if servers are busy
    try:
        response = client.models.generate_content(
            model='gemini-3.8-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        # Fallback to an alternate flash model if the primary one has a traffic spike
        print("[INFO] Primary model busy, trying fallback model...")
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
        )
        return response.text

if __name__ == "__main__":
    print("==================================================")
    print(" Interactive LLM Document Information Extractor")
    print("==================================================")
    print("Paste or type your document text below.")
    print("When you are finished entering your text, type the word 'END' on a new line and press Enter:\n")

    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break

    user_document = "\n".join(lines)

    if not user_document.strip():
        print("[ERROR] No text provided. Please run the script again.")
    else:
        print("\n[INFO] Sending document to Gemini for intelligent extraction...")
        try:
            structured_json = extract_with_gemini(user_document)
            print("\n--- Structured JSON Output ---")
            print(structured_json)
        except Exception as e:
            print(f"\n[ERROR] Execution failed. Details: {e}")