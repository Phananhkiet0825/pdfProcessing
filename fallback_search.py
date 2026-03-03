import os
import re

def extract_so_van_ban(pdf_filename):
    basename = os.path.splitext(pdf_filename)[0]
    numbers = re.findall(r'\d{1,4}', basename)
    if len(numbers) >= 2:
        return numbers[0], numbers[1]
    return None, None

def split_tokens(text):
    # Split by delimiters or non-alphanumeric chars
    return re.split(r'[\W_]+', text.lower())

def is_number_exact_in_tokens(number, tokens):
    # Check if number appears exactly in tokens list
    return number.lower() in tokens

def find_matching_txt_by_so(pdf_filename, text_folder):
    num1, num2 = extract_so_van_ban(pdf_filename)
    if not num1 or not num2:
        return None
    
    for fname in os.listdir(text_folder):
        fname_tokens = split_tokens(fname)
        if is_number_exact_in_tokens(num1, fname_tokens) and is_number_exact_in_tokens(num2, fname_tokens):
            return os.path.join(text_folder, fname)
    return None
