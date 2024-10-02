import pandas as pd
import numpy as np
import ast
import operator
import json
from fuzzywuzzy import fuzz
import os
from python_algo.semantic import Semantic_calculate
import time

class question_database_manage:
    def __init__(self, LLM, question_database_path):
        self.data_path = question_database_path
        self.data = pd.read_csv(question_database_path)
        self.LLM = LLM
    
    def add_question(self, question_content):
        new_row_df = pd.DataFrame([question_content])  # Convert question data to DataFrame
        self.data = pd.concat([self.data, new_row_df], ignore_index=True)
        self.data.to_csv(self.data_path, index=False)
    
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
    
    def transform_str_numpy_array(self, string_matrix):
        
        listMatrix = json.loads(string_matrix)
        numpy_array = np.array(listMatrix)
        
        return numpy_array
    
    def search_spatial(self, question_content):
        
        list_rank = {}
        for row in self.data.iloc:
           
            new_matrix = np.array(question_content['spatial_matrix'])
            old_matrix = self.transform_str_numpy_array(row['spatial_matrix'])
            list_rank[str(row['id'])] = self.compare_matrices(new_matrix, old_matrix)
        return list_rank
    
    
    def new_ranking_question(self, question_content, k= 20):
        spatial_result  = self.search_spatial(question_content)

        sorted_ranks = sorted(spatial_result.items(), key=operator.itemgetter(1), reverse=True)
        top_scores = []
        for i in range(0, k):
            top_scores.append(list(sorted_ranks[i]))

        for i in range(k, len(sorted_ranks)):
            if int(sorted_ranks[i][1]) == 1:
                top_scores.append(list(sorted_ranks[i]))  # Add ties to top_scores
            else:
                break  # Stop if scores are lower

        top_scores.sort(key=lambda x: x[1], reverse=True)
        new_top_score = []

        list_semantic = []
        for item in top_scores:
            item = list(item)
            item_id = item[0]
            compare_question = self.data[self.data['id'] == int(item_id)]
            list_semantic.append(compare_question['question_content'].iloc[0])
        semantic_result = Semantic_calculate(question_content, list_semantic)
        semantic_result = semantic_result.tolist()

        count = 0
        list_question=""
        input_question = {"id": question_content['id'], "question_content": question_content["question_content"], "ans": question_content["ans"], "instruction": question_content["instruction"]}
        for item in top_scores:
            item = list(item)
            compare_question_id = item[0]
            compare_question_content = self.data[self.data['id'] == int(compare_question_id)]['question_content'].iloc[0]
            compare_question_ans = self.data[self.data['id'] == int(compare_question_id)]["ans"].iloc[0]
            compare_question_ins = self.data[self.data['id'] == int(compare_question_id)]["instruction"].iloc[0]
            list_question += " [id: " + str(compare_question_id) + ", Content: " + str(compare_question_content) + ", Correct answer:" + str(compare_question_ans) + ", Instruction:" + str(compare_question_ins) + ", IoU score:" + str(item[1]) + ", Semantic score:" + str(semantic_result[0][count]) + "] "           
            count+=1
        prompt = self.LLM.get_prompt(2,input_question,list_question)

        answer = self.LLM.get_completion(prompt)
        print(answer)
        index=0
        answer = json.loads(answer)
        
        for id, details in answer.items():
            #search for item in top_scores that has the same id
            for i in range(len(top_scores)):
                if top_scores[i][0] == id:
                    index = i
            new_top_score.append([id, top_scores[index][1], semantic_result[0][index], int(details['Level']), details['Reason']])
            # new_top_score.append([id, top_scores[index][1], '70%', semantic_result[index], details['Reason']])

        new_top_score = sorted(new_top_score, key=lambda x: (-x[3], -x[1], -x[2]))
        return new_top_score, answer
    
    def process_questions(self,question_content_add, csv_file_path):
        df = pd.read_csv(csv_file_path)
        new_question = {
            "id": question_content_add["id"],
            "question_content": question_content_add["question_content"],
            "ans": question_content_add["ans"],
            "difficulty": question_content_add["difficulty"],
            "learning_outcome": question_content_add.get("learning_outcome", "LO3"),  # Default to "LO2" if not present
            "subchapters": question_content_add["subchapters"],
            "paragraph": question_content_add["paragraph"],
            "instruction": question_content_add["instruction"],
            "spatial_matrix": question_content_add["spatial_matrix"]
        }
      
        def handle_nan(value):
            if pd.isna(value):
                return "nan"
            return value
        def parse_subchapters(value):
            if pd.isna(value) or value == "nan":
                return ["nan"]
            try:
                # Try parsing the value as a list
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                # If it fails, return the value as a single-item list
                return [value]
        # Initialize the list for old questions
        old_questions = []
        for index, row in df.iterrows():
            old_question = {
                "id": handle_nan(row['id']),
                "question_content": handle_nan(row['question_content']),
                "ans": handle_nan(row['ans']),
                "difficulty": handle_nan(row['difficulty']),
                "learning_outcome": handle_nan(row['learning_outcome']),
                "subchapters": parse_subchapters(row["subchapters"]),
                "instruction": handle_nan((row['instruction']))
            }
            old_questions.append(old_question)
        result = {
            "old_question": old_questions,
            "new_question": new_question
        }
        return result
    
    def save_log(self, data, file_path = r"data\log_data\log_data2.json"):
        def append_to_json_file(log_data, file_path):
            """Appends log data to a JSON file with auto-incrementing IDs.

            Args:
                log_data (dict): The log data to append (without id_log).
                filename (str): The path to the JSON file (default: "log.json").
            """

            if not os.path.exists(file_path):
                # Create an empty list if the file is new
                with open(file_path, "w") as file:
                    json.dump([], file)

            with open(file_path, "r+") as file:
                file_data = json.load(file)

                # Ensure file_data is a list for appending
                if not isinstance(file_data, list):
                    raise ValueError(f"Invalid JSON structure in {file_data}. Expected a list.")

                # Determine the next ID
                if file_data:
                    last_id = max(entry["id_log"] for entry in file_data)
                    new_id = last_id + 1
                else:
                    new_id = 1
                
                # Add ID to the log data
                log_data["id_log"] = new_id
                
                # Append to the data
                file_data.append(log_data)

                file.seek(0)  
                json.dump(file_data, file, indent=4)
        
                
        append_to_json_file(data, file_path)
