from flask import Flask, redirect, url_for, render_template, request, send_from_directory, Response, jsonify
import pandas as pd
import os
import uuid
from python_algo.spatial_process import SpatialTransform
from python_algo.database_management import question_database_manage
from python_algo.data_plot import data_analyze
from python_algo.gemini_api import LLM
import json
import csv
import numpy as np
from python_algo.statistic import statistic
import time

global question_data, subchapter_spatial
pdf_path = r"current\input_file\pythonlearn.pdf"
database_path = r"data\duplicate_matrix1.csv"
new_question_path = r"data\question_data.csv"
log_path = r"data\log_data\log_data2.json"

#gemini_call = None
gemini_call = LLM()
transformer = SpatialTransform(pdf_path)
question_database = question_database_manage(gemini_call, database_path)
plot_graph = data_analyze(database_path)
report = statistic(log_path)


question_data = pd.read_csv(new_question_path)
subchapter_spatial = transformer.create_subchapter_matrix(pdf_path)


app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('web/viewer.html')

@app.route('/web/<path:filename>')
def custom_static(filename):
    if filename.endswith('.mjs'):
        file_path = os.path.join(app.root_path, 'templates/web', filename)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return Response(file_content, mimetype='application/javascript')
    else:
        return send_from_directory(os.path.join(app.root_path, 'templates/web'), filename)

@app.route('/build/<path:filename>')
def serve_build(filename):
    if filename.endswith('.mjs'):
        file_path = os.path.join(app.root_path, 'templates/build', filename)
        with open(file_path, 'rb') as f:
            file_content = f.read()
        return Response(file_content, mimetype='application/javascript')
    else:
        return send_from_directory(os.path.join(app.root_path, 'templates/build'), filename)


UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        questions_data = []

        # Đọc tệp CSV và lấy câu hỏi đầu tiên
        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = [row for row in reader if any(row)]
            
            for row in rows[1:]:
                question_data = {
                    'id': row[0],
                    'question': row[1],
                    'answer': row[2],
                    'difficulty': None,
                }
                questions_data.append(question_data)

        return jsonify({'data':questions_data, 'success': True})


@app.route("/submit", methods=["POST"])
def submit():
    collected_data = request.form.get("collectedData")
  
    #spatial transform from paragraph
    matrix = transformer.spatial_return(collected_data) 
  
    #extract table of content related
    table_content = transformer.get_subchapters_from_fragments(subchapter_spatial ,matrix)
    
    graph_data = plot_graph.plot(table_content, matrix)
    filename = r"current\graph.json"

    # Open the file in write mode ('w') and save as JSON
    with open(filename, 'w', encoding='utf-8') as outfile:
        json.dump(graph_data, outfile, ensure_ascii=False, indent=4)
    return jsonify({'data': 'success'})
    #return render_template('plot.html', question_content=collected_data)


@app.route("/submitquestion", methods=["POST"])
def submitquestion():
    collected_data = request.form.get("collectedData")
    #extract question from upload file and compose information
    question_id = json.loads(collected_data)["id"]
    # print("Hello", question_data)
    question_content = question_data[question_data["id"] == int(question_id)]
    #spatial transform from paragraph
    matrix = transformer.spatial_return(collected_data) 
  
    #extract table of content related
    table_content = transformer.get_subchapters_from_fragments(subchapter_spatial ,matrix)
    # instruction = None
    instruc_prompt = gemini_call.get_prompt(1, {"id": question_id, "question_content": question_content["question_content"].iloc[0], "ans": question_content["ans"].iloc[0]}, None, collected_data)
    
    instruction = gemini_call.get_completion(instruc_prompt)
    try:
        instruction = json.loads(instruction)
        instruction = instruction["Correct answer explanation"]
    except json.JSONDecodeError as e:
        instruction = instruction
    
    
    question_content_add = {"id": question_id, "question_content": question_content["question_content"].iloc[0], "ans": question_content["ans"].iloc[0], "difficulty":  int(question_content['difficulty'].iloc[0]),"subchapters": table_content, "paragraph": collected_data,"instruction": instruction,"spatial_matrix": matrix.tolist()}
    question_content_log = {"id": question_id, "question_content": question_content["question_content"].iloc[0], "ans": question_content["ans"].iloc[0], "difficulty": int(question_content['difficulty'].iloc[0]),"subchapters": table_content, "paragraph": collected_data,"instruction": instruction}


    ranking_result, answer = question_database.new_ranking_question(question_content_add)
    
    question_compare = question_database.process_questions(question_content_add, database_path)

    filename = r"current\graph.json"
    with open(filename, 'r', encoding='utf-8') as infile:
        graph_data =  json.load(infile)
    
    log_data = {"id_log": None, "question_input": question_content_log, "ranking_result": ranking_result, 'explain': answer, 'graph': graph_data, 'question_compare': question_compare}
    question_database.save_log(log_data)

 
    return jsonify({'data': ranking_result, 'question_compare': question_compare, 'success': True})



@app.route('/plot')
def index():
    return render_template('plot.html')

@app.route('/data')
def data():
    filename = r"current\graph.json"
    with open(filename, 'r', encoding='utf-8') as infile:
        graph_data =  json.load(infile)
  
    data = convert_data(graph_data)
 
    return jsonify(data)



@app.route('/getLog', methods=["POST"])
def getLog():
    filename = r"data\log_data\log_data2.json"
    with open(filename, 'r', encoding='utf-8') as infile:
            data =  json.load(infile)
    return jsonify({'data': data})


def convert_data(input_data):
    output_data = {}
    
    if 'first_graph' in input_data:
        output_data['chart1'] = {
            'learning_outcome': input_data['first_graph'].get('learning_outcome', []),
            'number': input_data['first_graph'].get('number', [])
        }

    if 'second_graph' in input_data:
        output_data['chart2'] = {
            'subchapter': input_data['second_graph'].get('subchapter', []),
            'number': input_data['second_graph'].get('number', [])
        }

    if 'third_graph' in input_data:
        output_data['chart3'] = {
            'subchapter': input_data['third_graph'].get('subchapter', []),
            'difficult_level': input_data['third_graph'].get('number', [])
        }

    
    if 'four_graph' in input_data:
        output_data['chart4'] = {
            'spatial_match': input_data['four_graph'].get('spatial_match', []),
            'number': input_data['four_graph'].get('number', [])
        }
        
    return output_data


@app.route('/getSubChapter', methods=['POST'])
def getSubChapter():
    data = {
        "learning_outcome": ["LO1", "LO2", "LO3", "LO4", "LO5", "LO6", "LO7"],
        "subchapters": [
            ["1.2", "1.3", "1.4", "1.6"],             # LO1
            ["2.2", "2.3", "2.4", "2.6"],             # LO2
            ["3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9",  # LO3
             "4.1", "4.2", "4.3", "4.4", "4.5", "4.6", "4.7", "4.8", "4.9", "4.10", "4.11", "4.12", 
             "5.1", "5.2", "5.3", "5.4", "5.5", "5.6", "5.7"], # LO3
            ["6.1", "6.2", "6.3", "6.4", "6.5", "6.6", "6.7", "6.8", "6.9", "6.10", "6.11", "6.12"], # LO4
            ["7.1", "7.2", "7.3", "7.4", "7.5", "7.6", "7.7", "7.8", "7.9"], # LO5
            ["8.1", "8.2", "8.3", "8.4", "8.5", "8.6", "8.7", "8.8", "8.9", "8.10", "8.11", "8.12",  # LO6
             "9.1", "9.2", "9.3", "9.4", "9.5", "10.1", "10.2", "10.3", "10.4"], 
            ["14.1", "14.2", "14.3", "14.4", "14.5", "14.6", "14.7", "14.8", "14.9", "14.10", "14.11"] # LO7
        ]
    }
    subchapters = data["subchapters"]
    learning_outcome = data["learning_outcome"]
    return jsonify({"subchapters": subchapters, "learning_outcome": learning_outcome})


@app.route('/formatDataChart', methods=['POST'])
def formatDataChart():
    data = request.form.get("data")
    print("raw",data)
    print("convert",log_convert_data(data))
    return jsonify(log_convert_data(data))

def log_convert_data(input_data):
    output_data = {}
    input_data = json.loads(input_data)
    if 'first_graph' in input_data:
        output_data['chart1'] = {
            'learning_outcome': input_data['first_graph'].get('learning_outcome', []),
            'number': input_data['first_graph'].get('number', [])
        }

    if 'second_graph' in input_data:
        output_data['chart2'] = {
            'subchapter': input_data['second_graph'].get('subchapter', []),
            'number': input_data['second_graph'].get('number', [])
        }

    if 'third_graph' in input_data:
        output_data['chart3'] = {
            'subchapter': input_data['third_graph'].get('subchapter', []),
            'difficult_level': input_data['third_graph'].get('number', [])
        }

    if 'four_graph' in input_data:
        output_data['chart4'] = {
            'spatial_match': input_data['four_graph'].get('spatial_match', []),
            'number': input_data['four_graph'].get('number', [])
        }
    return output_data

@app.route('/save_question', methods=['POST'])
def save_question():
    try:
        data = request.get_json()  # Nhận dữ liệu JSON từ request
        new_question = data.get('new_question')

        # Lưu dữ liệu vào file CSV
        with open('data/questions.csv', 'a', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['id', 'question_content', 'ans', 'difficulty', 'learning_outcome', 
                          'subchapters', 'paragraph', 'instruction', 'spatial_matrix']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # Kiểm tra xem file CSV có trống không
            # Nếu trống thì ghi header
            if csvfile.tell() == 0:
                writer.writeheader()
            writer.writerow(new_question)  

        return jsonify({'message': 'Success!'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

