import fitz
import os
import cv2
import numpy as np
import tensorflow as tf
import shutil
import re
import pdfplumber
from PIL import Image
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

# Sử dụng đường dẫn tương đối
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "classifier", "imageclassifier2.h5")

def call_groq(prompt):
    """Gọi Groq API để xử lý văn bản"""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    chat = ChatGroq(
        api_key=groq_api_key,
        model="llama3-70b-8192",
        temperature=0
    )
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = chat.invoke(messages)
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        return ""
    except Exception as e:
        print(f"Error calling Groq API: {e}")
        return ""

def get_book_name(pdf_path):
    """Trích xuất tên sách và tác giả từ PDF"""
    def detect_toc(text):
        prompt = (
            "You are an AI assistant. I will provide a text passage, please identify and extract only the Title and Author from it. "
            "Do not include any other content. "
            "Respond strictly in the following format: name of the book by author of the book. "
            "If no title or author is found, say 'No title or author found'. "
            f"Text passage:\n{text}"
        )
        return call_groq(prompt)

    try:
        with pdfplumber.open(pdf_path) as pdf:
            extracted_texts = ""
            # Chỉ xử lý 5 trang đầu
            for page in pdf.pages[:5]:
                text = page.extract_text()
                if text:
                    extracted_texts += text + "\n"
            
            sample = detect_toc(extracted_texts)
            return sample if sample else "No title or author found"
    except Exception as e:
        print(f"Error extracting book name: {e}")
        return "No title or author found"

def extract_contents_from_pdf(pdf_path, model_path=None):
    """Trích xuất mục lục từ PDF - chỉ xử lý 30 trang đầu"""
    if model_path is None:
        model_path = MODEL_PATH
    
    def detect_toc(text):
        prompt = (
            "You are an AI assistant. I will provide a text passage, "
            "please identify and extract only the main chapter headings (e.g., 1, 2, 3) "
            "and their subheadings (e.g., 1.1, 1.2, 1.3) along with their associated page numbers if available. "
            "Do not include any other content. "
            "Respond in a structured format listing only the relevant headings and subheadings. "
            "If no table of contents is found, say 'No table of contents found'.\n\n"
            "Do not use character '-' in the response."
            f"Text passage:\n{text}"
        )
        return call_groq(prompt)
    
    # Tạo thư mục tạm
    correct_dir = os.path.join(BASE_DIR, "temp", "correct")
    os.makedirs(correct_dir, exist_ok=True)

    try:
        print(f"Processing first 30 pages of PDF: {pdf_path}")
        
        # Bước 1: Chuyển đổi 30 trang đầu thành hình ảnh
        import fitz
        doc = fitz.open(pdf_path)
        total_pages = min(30, doc.page_count)
        
        print(f"Converting {total_pages} pages to images...")
        for i in range(total_pages):
            page = doc.load_page(i)
            pix = page.get_pixmap()
            pix.save(f"{correct_dir}/{i}.png")

        # Bước 2: Sử dụng model để lọc các trang có table of contents (nếu model tồn tại)
        toc_pages = []
        if os.path.exists(model_path):
            print("Using ML model to detect TOC pages...")
            model = tf.keras.models.load_model(model_path)
            image_files = [f for f in os.listdir(correct_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in image_files:
                img_path = os.path.join(correct_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Cannot read {img_name}")
                    continue
                
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
                img_resized = tf.image.resize(img_tensor, (256, 256))
                img_input = np.expand_dims(img_resized / 255.0, axis=0)
                yhat = model.predict(img_input, verbose=0)
                
                # Nếu model predict là TOC page
                if yhat[0][0] <= 0.5:  # <= 0.5 là TOC page
                    page_num = int(img_name.split('.')[0])
                    toc_pages.append(page_num)
            
            print(f"Model detected TOC pages: {toc_pages}")
        else:
            # Nếu không có model, sử dụng keyword search trong 30 trang đầu
            print("No ML model found, using keyword search...")
            with pdfplumber.open(pdf_path) as pdf:
                toc_keywords = ["table of contents", "contents", "mục lục", "chapter", "chương"]
                
                for i in range(min(30, len(pdf.pages))):
                    page = pdf.pages[i]
                    text = page.extract_text()
                    if text:
                        text_lower = text.lower()
                        # Kiểm tra từ khóa mục lục
                        has_toc_keyword = any(keyword in text_lower for keyword in toc_keywords)
                        
                        # Kiểm tra pattern số trang
                        page_pattern = re.findall(r'\d+\.{2,}\d+|\d+\s+\d+\s+\d+', text)
                        has_page_numbers = len(page_pattern) > 3
                        
                        if has_toc_keyword or has_page_numbers:
                            toc_pages.append(i)
                            print(f"Found potential TOC at page {i + 1}")

        # Nếu không tìm thấy TOC pages, sử dụng 10 trang đầu
        if not toc_pages:
            toc_pages = list(range(min(10, total_pages)))
            print("No TOC detected, using first 10 pages as fallback")

        # Bước 3: Trích xuất văn bản từ các trang TOC bằng pdfplumber
        print(f"Extracting text from {len(toc_pages)} TOC pages...")
        extracted_texts = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page_num in toc_pages:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    text = page.extract_text()
                    if text and text.strip():
                        extracted_texts += f"\n--- Page {page_num + 1} ---\n{text}\n"

        # Bước 4: Xử lý văn bản
        def merge_numeric_lines(text):
            if not text:
                return ""
                
            lines = text.split('\n')
            merged_lines = []
            pattern_number = re.compile(r'\d+$')
            pattern_section = re.compile(r'\d+\.\d+')
            pattern_subsection = re.compile(r'\d+\.\d+\.\d+')
            pattern_chapter = re.compile(r'Chapter \d+$')
            pattern_CHAPTER = re.compile(r'CHAPTER \d+$')
            
            remove_keywords = {
                "CONTENTS", "Contents", "Table of Contents", "Preface", "preface", "PREFACE", 
                "page ", "Page", "Online Resources", "ONLINE RESOURCES", 
                "About the Author", "ABOUT THE AUTHOR", "ISBN", "Copyright"
            }
            
            filtered_lines = [line for line in lines if not any(keyword in line for keyword in remove_keywords)]
            
            for i, line in enumerate(filtered_lines):
                line_stripped = line.strip()
                if pattern_number.fullmatch(line_stripped):
                    if merged_lines:
                        merged_lines[-1] += ' ' + line_stripped
                elif pattern_section.fullmatch(line_stripped) or pattern_subsection.fullmatch(line_stripped):
                    merged_lines.append(line)
                elif (i > 0 and 
                      (pattern_section.fullmatch(filtered_lines[i - 1].strip()) or 
                       pattern_subsection.fullmatch(filtered_lines[i - 1].strip()))):
                    merged_lines[-1] += ' ' + line_stripped
                else:
                    merged_lines.append(line)
            
            refined_lines = [re.sub(r'(\d+) (\d+)$', r'\1', line) for line in merged_lines]
            
            final_lines = []
            i = 0
            while i < len(refined_lines):
                if ((pattern_chapter.fullmatch(refined_lines[i].strip()) or 
                     pattern_CHAPTER.fullmatch(refined_lines[i].strip())) and 
                    i + 1 < len(refined_lines)):
                    final_lines.append(refined_lines[i] + ' ' + refined_lines[i + 1])
                    i += 2
                else:
                    final_lines.append(refined_lines[i])
                    i += 1
            
            final_lines = [re.sub(r'\.$', '', line) for line in final_lines]
            return '\n'.join(final_lines)

        def clean_text(text):
            if not text:
                return ""
            lines = text.split("\n")
            cleaned_lines = [line.strip() for line in lines if line.strip()]
            return "\n".join(cleaned_lines)
        
        # Xử lý văn bản được trích xuất
        print("Processing extracted text...")
        processed_text = merge_numeric_lines(extracted_texts)
        
        print("Calling Groq API for TOC extraction...")
        sample = detect_toc(processed_text)
        
        result = clean_text(sample)
        
        print(f"TOC extraction completed. Found {len(result.split('\n')) if result else 0} lines of content.")
        return result if result else "No table of contents found"
        
    except Exception as e:
        print(f"Error extracting contents: {e}")
        import traceback
        traceback.print_exc()
        return "Error extracting table of contents"
    finally:
        # Cleanup thư mục tạm
        if os.path.exists(correct_dir):
            shutil.rmtree(correct_dir, ignore_errors=True)

def process_pdf(pdf_path):
    """Xử lý PDF để lấy thông tin sách và mục lục"""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
            
        print(f"Processing PDF: {pdf_path}")
        
        book_info = get_book_name(pdf_path)
        content = extract_contents_from_pdf(pdf_path)
        
        result = {
            "book": book_info,
            "content": content
        }
        
        print("PDF processing completed successfully")
        return result
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {
            "book": "Error extracting book information",
            "content": "Error extracting content"
        }

def test(pdf_path):
    """Hàm test để kiểm tra xử lý PDF"""
    try:
        book_info = get_book_name(pdf_path)
        content = extract_contents_from_pdf(pdf_path)
        print(f"Title: {book_info}")
        print(f"Content: {content}")
        return {
            "book": book_info,
            "content": content
        }
    except Exception as e:
        print(f"Test failed: {e}")
        return None