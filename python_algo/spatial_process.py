import fitz  # PyMuPDF
import numpy as np
import csv
import json
from collections import Counter
from fuzzywuzzy import fuzz

class SpatialTransform:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.text_by_page = self.extract_text_from_pdf()
        
    def celi(self,x):
        if x == int(x):
            return int(x)
        else:
            return int(x)+1
        
    def custom_round(self, x):
        if x%1 >= 0.5:
            return int(x)+1
        else:
            return int(x)

    def extract_text_from_pdf(self):
        doc = fitz.open(self.pdf_path)
        text_by_page = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text").replace('\n', ' ')
            text_by_page.append(text)
        return text_by_page
    
   
    
    def convert_toc(self, old_toc):
        new_toc = []
        section_counters = [0]

        for level, title, page_num in old_toc:
            # Ensure the section_counters list is long enough for the current level
            while len(section_counters) < level:
                section_counters.append(0)  # Initialize new levels with 0

            # Reset counters for deeper levels when moving to a higher level
            section_counters = section_counters[:level]  

            # Increment the counter for the current level
            section_counters[level - 1] += 1 

            # Construct the full section number
            section_num = ".".join(str(x) for x in section_counters)

            new_entry = [section_num, title, page_num]
            new_toc.append(new_entry)

   
        modify_toc  = [item for item in new_toc if len(item[0]) != 1]
        return modify_toc

    def modify_toc(self, link_pdf):
        doc = fitz.open(link_pdf)
        text_by_page = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text").replace('\n', ' ')
            text_by_page.append(text)
            
        old_toc = doc.get_toc()
        return self.convert_toc(old_toc)
    
    def create_subchapter_matrix(self, link_pdf):
        doc = fitz.open(link_pdf)
        text_by_page = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text = page.get_text("text").replace('\n', ' ')
            text_by_page.append(text)
            
        old_toc = doc.get_toc()
        modify_toc = self.convert_toc(old_toc)
    
            
        n = len(text_by_page)  # number of pages
        m = 100  # number of columns representing positions in a page
        subchapter_matrix = np.empty((n, m), dtype=object)
        start_pos = 0
        start_page = 13 #page bắt đầu của co đầu tiên
        level_before = modify_toc[0][0]

        for chapter_info in modify_toc:
            level, title, end_page = chapter_info
            subchapter_id = f"{level} {title}"
            if end_page > len(text_by_page):
                continue
            text = text_by_page[end_page-1]
            
            # Find the start and end indices of the subchapter
            end_idx = text.find(subchapter_id)
            if end_idx == -1:
                continue
            # if start_idx == 0:
            #     start_idx = end_idx
            #     continue
            
            total_chars = len(text)
            end_pos = int((end_idx / total_chars) * m)
            
            if end_page > start_page:
                subchapter_matrix[start_page-1, start_pos:] = level_before
                for i in range (start_page+1, end_page):
                    subchapter_matrix[i-1, :] = level_before
                subchapter_matrix[end_page-1, :min(end_pos + 1, m)] = level_before
                start_pos = end_pos+1
                start_page = end_page
                level_before = level
            else:
                subchapter_matrix[end_page-1, start_pos:min(end_pos + 1, m)] = level_before
                start_pos = end_pos+1
                start_page = end_page
                level_before = level
        
        return subchapter_matrix
    
    def extract_fragments_from_json(self, json_data):
        fragments = []
        data = json.loads(json_data)

        for note in data.get("notes", []):  # Handle cases where "notes" might be missing
            if "knowledge" in note and "page" in note:
                fragments.append((note["knowledge"], int(note["page"])))

        return fragments
    
    
    def get_subchapters_from_fragments(self, subchapter_matrix,matrix):
        fragment_matrix = matrix
        subchapters = []
        
        for i in range(fragment_matrix.shape[0]):
            for j in range(fragment_matrix.shape[1]):
                if fragment_matrix[i, j] == 1:
                    subchapters.append(subchapter_matrix[i,j])
        element_counts = Counter(subchapters)
        # Lấy những phần tử xuất hiện từ 2 lần trở lên
        result = [element for element, count in element_counts.items() if count >= 2]
        return result
    
    def find_chunk_by_splitting(self, chunk, long_text):
        chunk_length = len(chunk)
        best_match = None
        best_ratio = 0

        # Split long_text into chunks of the same length
        chunks = [long_text[i:i + chunk_length] for i in range(0, len(long_text) - chunk_length + 1)]
        
        # Find the best matching chunk
        for i, current_chunk in enumerate(chunks):
            ratio = fuzz.ratio(chunk, current_chunk)
            if ratio > best_ratio:
                best_match = current_chunk
                best_ratio = ratio

        # Find the start and end positions in long_text
        if best_match:
            start_position = long_text.find(best_match)
            end_position = start_position + chunk_length
            return start_position, end_position
        else:
            return None, None
        
        
    
    def create_2d_matrix(self, data):
        self.data = data  # Store the input data directly
        self.fragments = []  # Initialize fragments list

        # Determine input type and extract fragments accordingly
        if isinstance(data[0], str): 
            self.fragments = self.extract_fragments_from_json(data)
        else:  
            self.fragments = data
        
        print(self.fragments)
        
        n = 245  # number of pages
        m = 100  # number of columns representing positions in a page
        matrix = np.zeros((n, m), dtype=int)
        
        for fragment, page in self.fragments:
            if page > len(self.text_by_page):
                print('Out of number page')
                continue
            text = self.text_by_page[page-1]
            
            start_idx, end_idx = self.find_chunk_by_splitting(fragment, text)
            
            total_chars = len(text)
            start_pos = int((start_idx / total_chars) * m)
            end_pos = self.celi((end_idx / total_chars) * m)
            
            matrix[page-1, start_pos:min(end_pos + 1, m)] = 1
        
        return matrix
    
    def spatial_return(self, data):
        return self.create_2d_matrix(data)
    
    def save_multiple_matrices_csv(self, file_path):
        with open(file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(self.create_2d_matrix())
            writer.writerow(["---"])
            
    def calculate_iou(self, matrix1, matrix2):
        intersection = np.logical_and(matrix1, matrix2)
        union = np.logical_or(matrix1, matrix2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def get_rows_with_non_zero(self, matrix):
        return np.where(np.any(matrix != 0, axis=1))[0]

    def compare_matrices(self, matrix1, matrix2):
        # Get rows with non-zero elements in both matrices
        rows_matrix1 = self.get_rows_with_non_zero(matrix1)
        rows_matrix2 = self.get_rows_with_non_zero(matrix2)
        
        # Union of both row indices
        rows_to_take = np.union1d(rows_matrix1, rows_matrix2)
        
        # Extract the relevant rows from both matrices
        matrix1_small = matrix1[rows_to_take]
        matrix2_small = matrix2[rows_to_take]
        
        # Calculate IOU between the two small matrices
        iou = self.calculate_iou(matrix1_small, matrix2_small)
        
        return iou

