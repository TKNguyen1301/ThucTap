import pdfplumber
import os
import re
import fitz
import cv2
import numpy as np
import tensorflow as tf
from shutil import rmtree
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Sử dụng đường dẫn tương đối từ gốc dự án
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "classifier", "model.h5")

def call_groq(prompt):
    import time
    
    # Sử dụng API key từ environment variable thay vì hardcode
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in environment variables")
    
    chat = ChatGroq(
        api_key=groq_api_key,  
        model="llama3-70b-8192",
        temperature=0,
        max_tokens=4000
    )
    messages = [
        {"role": "system", "content": "You are an AI assistant specialized in extracting table of contents from documents. Always provide complete and detailed responses."},
        {"role": "user", "content": prompt}
    ]
    
    time.sleep(0.5)
    
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
            "Do not include any other content. Respond strictly in the following format: name of the book by author of the book. "
            "If no title or author is found, say 'No title or author found'."
            f"Text passage:\n{text}"
        )
        return call_groq(prompt)

    extracted_texts = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Chỉ xử lý 5 trang đầu để tìm thông tin sách
            for page in pdf.pages[:5]:
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    extracted_texts += text + "\n"
    except Exception as e:
        print(f"Không thể trích xuất văn bản bằng pdfplumber: {e}")
        return "No title or author found"

    if not extracted_texts.strip():
        return "No title or author found"

    sample = detect_toc(extracted_texts)
    return sample if sample else "No title or author found"

def extract_contents_from_pdf(pdf_path, model_path=None):
    """Trích xuất mục lục từ PDF"""
    if model_path is None:
        model_path = MODEL_PATH
    
    def detect_toc_chunked(text):
        import time
        
        if not text or not text.strip():
            return ""
            
        text_chunks = []
        lines = text.split('\n')
        chunk_size = 200
        
        for i in range(0, len(lines), chunk_size):
            chunk = '\n'.join(lines[i:i+chunk_size])
            if len(chunk.strip()) > 0:
                text_chunks.append(chunk)
        
        if not text_chunks:
            return ""
            
        all_results = []
        for i, chunk in enumerate(text_chunks):
            print(f"Processing chunk {i+1}/{len(text_chunks)}")
            prompt = (
                "Extract ALL table of contents entries from this text chunk. "
                "DO NOT summarize, truncate, or omit ANY entries. "
                "Extract EVERY chapter, section, and subsection you find. "
                "Include ALL numbering levels (1, 1.1, 1.1.1, etc.) with page numbers. "
                "DO NOT add phrases like 'omitted for brevity' or similar. "
                "If this chunk contains no table of contents, respond with 'NONE'. "
                f"Text:\n{chunk[:4000]}"
            )
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    result = call_groq(prompt)
                    if result and result.strip():
                        if "NONE" not in result.upper():
                            all_results.append(result.strip())
                            print(f"Extracted content from chunk {i+1}")
                        else:
                            print(f"No TOC content in chunk {i+1}")
                        break
                except Exception as e:
                    print(f"Error processing chunk {i+1}, retry {retry+1}: {e}")
                    if retry < max_retries - 1:
                        time.sleep(2 ** retry)
                    else:
                        print(f"Failed to process chunk {i+1} after {max_retries} retries")
                        continue
        
        combined_result = '\n'.join(all_results)
        print(f"Total chunks processed: {len(all_results)} out of {len(text_chunks)}")
        return combined_result

    def merge_numeric_lines(text):
        if not text:
            return ""
            
        lines = text.split('\n')
        merged_lines = []
        pattern_number = re.compile(r'\d+$')
        pattern_section = re.compile(r'\d+\.\d+$')
        pattern_subsection = re.compile(r'\d+\.\d+\.\d+$')
        pattern_subsubsection = re.compile(r'\d+\.\d+\.\d+\.\d+$')
        pattern_chapter = re.compile(r'Chapter \d+$')
        pattern_CHAPTER = re.compile(r'CHAPTER \d+$')
        
        remove_keywords = {"CONTENTS", "Contents", "Table of Contents", "Preface", "preface", "PREFACE",
                          "page ", "Page", "Online Resources", "ONLINE RESOURCES", "About the Author", "ABOUT THE AUTHOR"}
        filtered_lines = [line for line in lines if not any(keyword in line for keyword in remove_keywords)]
        
        for i, line in enumerate(filtered_lines):
            line_stripped = line.strip()
            if pattern_number.fullmatch(line_stripped):
                if merged_lines:
                    merged_lines[-1] += ' ' + line_stripped
            elif (pattern_section.fullmatch(line_stripped) or 
                  pattern_subsection.fullmatch(line_stripped) or 
                  pattern_subsubsection.fullmatch(line_stripped)):
                merged_lines.append(line)
            elif (i > 0 and 
                  (pattern_section.fullmatch(filtered_lines[i-1].strip()) or 
                   pattern_subsection.fullmatch(filtered_lines[i-1].strip()) or
                   pattern_subsubsection.fullmatch(filtered_lines[i-1].strip()))):
                merged_lines[-1] += ' ' + line_stripped
            else:
                merged_lines.append(line)
        
        refined_lines = merged_lines
        refined_lines = [re.sub(r'(\d+(?:\.\d+)*) (\d+)$', r'\1', line) for line in refined_lines]
        
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
        cleaned_lines = [line for line in lines if len(line.strip()) > 1]
        return "\n".join(cleaned_lines)

    # Kiểm tra file PDF tồn tại
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    all_extracted_text = []
    use_ocr = False
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages in PDF: {total_pages}")
            print("Processing all pages with pdfplumber...")
            
            for page in pdf.pages:
                text = page.extract_text()
                if text and len(text.strip()) > 10:
                    all_extracted_text.append(f"--- Page {page.page_number} ---\n{text}")
                if page.page_number % 50 == 0:
                    print(f"Processed page {page.page_number}/{total_pages}")
            print(f"Completed pdfplumber extraction: {len(all_extracted_text)} pages")
            
    except Exception as e:
        print(f"Không thể trích xuất văn bản bằng pdfplumber: {e}. Chuyển sang OCR...")
        use_ocr = True

    if not all_extracted_text:
        use_ocr = True

    # Xử lý OCR nếu cần
    if use_ocr:
        # Tạo thư mục tạm trong thư mục dự án
        temp_dir = os.path.join(BASE_DIR, "temp")
        correct_dir = os.path.join(temp_dir, "correct")
        os.makedirs(correct_dir, exist_ok=True)

        try:
            doc = fitz.open(pdf_path)
            total_pages = doc.page_count
            print(f"Total pages in PDF: {total_pages}")
            print("Converting all pages to images...")
            
            for i in range(doc.page_count):
                page = doc.load_page(i)
                pix = page.get_pixmap()
                pix.save(f"{correct_dir}/{i}.png")
                if (i + 1) % 100 == 0:
                    print(f"Converted {i + 1}/{total_pages} pages to images")

            # Chỉ sử dụng classification model nếu file tồn tại
            if os.path.exists(model_path):
                print("Running classification model on all pages...")
                model = tf.keras.models.load_model(model_path)
                image_files = [f for f in os.listdir(correct_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_name in image_files:
                    img_path = os.path.join(correct_dir, img_name)
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Không thể đọc {img_name}")
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
                    img_resized = tf.image.resize(img_tensor, (256, 256))
                    img_input = np.expand_dims(img_resized / 255.0, axis=0)
                    yhat = model.predict(img_input)
                    if yhat[0][0] > 0.5:
                        os.remove(img_path)
                        
            # Cleanup
            rmtree(correct_dir, ignore_errors=True)
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            # Cleanup in case of error
            rmtree(correct_dir, ignore_errors=True)

    if not all_extracted_text:
        return "No content extracted from PDF"

    print(f"Total extracted text length: {len(all_extracted_text)} pages")
    
    # Tạo file tạm để debug (tùy chọn)
    try:
        full_text = '\n'.join(all_extracted_text)
        print("Starting text preprocessing...")
        processed_text = merge_numeric_lines(full_text)
        
        print("Starting LLM processing for table of contents extraction...")
        sample = detect_toc_chunked(processed_text)
        
        print("Cleaning extracted content...")
        result = clean_text(sample)
        
        print(f"Total content length: {len(result)} characters")
        print(f"Total lines: {len(result.split('\n'))}")
        
        return result if result else "No table of contents found"
        
    except Exception as e:
        print(f"Error in content processing: {e}")
        return "Error processing PDF content"

def process_pdf(pdf_path):
    """
    Xử lý PDF để lấy thông tin sách và mục lục
    
    Args:
        pdf_path: Đường dẫn đến file PDF
        
    Returns:
        dict: {"book": tên sách, "content": mục lục}
    """
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

def test_pdf_processing(pdf_path):
    """Hàm test để kiểm tra xử lý PDF"""
    try:
        result = process_pdf(pdf_path)
        print(f"Title: {result['book']}")
        print(f"Content: {result['content'][:500]}...")  # Chỉ hiển thị 500 ký tự đầu
        return result
    except Exception as e:
        print(f"Test failed: {e}")
        return None

# Để tương thích với code cũ
def test(pdf_path):
    """Hàm test tương thích với code cũ"""
    return test_pdf_processing(pdf_path)