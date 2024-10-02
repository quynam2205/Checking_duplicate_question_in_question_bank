import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import json
import ast

class data_analyze:
    def __init__(self, database_path = r"data\question_data2.csv"):
        self.database_path = database_path
   
        self.df = pd.read_csv(database_path)
    
    def calculate_iou(self, matrix1, matrix2):
        intersection = np.logical_and(matrix1, matrix2)
        union = np.logical_or(matrix1, matrix2)
        iou = np.sum(intersection) / np.sum(union)
        return iou

    def plot(self, subchapters, matrix):

        with open(r'data\listLo.json', 'r') as file:
            data_lo = json.load(file)
        subchapters = ['CO'+i for i in subchapters]

        def modify_subchapters(subchapters):
            if isinstance(subchapters, str):
                subchapters_list = eval(subchapters)  # Chuyển đổi từ chuỗi thành danh sách
                co_list = [f'CO{item}' for item in subchapters_list]
                return ', '.join(co_list)
            return ''

        
        if 'subchapters_change' not in self.df.columns:
            self.df['subchapters_change'] = self.df['subchapters'].apply(modify_subchapters)

        
        x = ['learning_outcome', 'subchapter', 'subchapter', 'spatial_match']
        y = ['number', 'number', 'number', 'number']
        name_graph = ['first_graph','second_graph', 'third_graph', 'four_graph']

        # GRAPH 1
        lo_graph1 = {}
        for i in data_lo.keys():
            lo_graph1[i] = 0
        
        for i in self.df['subchapters_change']:
            subchapter_g1 =[element.strip() for element in i.split(',')]
            memory = []
            memory_key = ""

            for i in subchapter_g1:
                index = i.find('.')+1
                for key, value in data_lo.items():
                    if i[:index] in value and key not in memory:
                        memory_key = key
                        memory.append(key)
            if memory_key in lo_graph1:
                lo_graph1[memory_key] += 1

        def form(name_graph, x , y):
            return {name_graph: {x: [], y:[]}}
        graph1 = form(name_graph[0], x[0], y[0])
        for iteam, count in lo_graph1.items():
            if 'LO' in iteam:
                graph1[name_graph[0]][x[0]].append(iteam)
                graph1[name_graph[0]][y[0]].append(count)

        #PLOT CO (GRAPH 2)
        all_lo = []
        co_all_counts = self.df['subchapters_change'].str.split(',').explode().str.strip().value_counts()
        for i in subchapters:
            i = str(i)
            index = i.find('.')+1
            for key, value in data_lo.items():
                if i[:index] in value and key not in all_lo:
                    all_lo.append(key)

        graph2 = form(name_graph[1], x[1], y[1])

        lo_numbers = [int(lo[2:]) for lo in all_lo]
        highest_lo = f"LO{max(lo_numbers)}"


        for item,count in co_all_counts.items():
            
            for co in data_lo[highest_lo]:
                if co in item:
                    
                    graph2[name_graph[1]][x[1]].append(item)
                    graph2[name_graph[1]][y[1]].append(count)

        # Sample lists
        co_list = graph2[name_graph[1]][x[1]]
        question_count_list = graph2[name_graph[1]][y[1]]

        # Function to convert "co" strings into tuples of integers for sorting
        def co_key(co_string):
            return tuple(map(int, co_string[2:].split('.')))

        # Combine the lists into pairs, sort by the co list using the custom key function
        sorted_pairs = sorted(zip(co_list, question_count_list), key=lambda pair: co_key(pair[0]))

        # Unzip the pairs back into two lists
        sorted_co_list, sorted_question_count_list = zip(*sorted_pairs)

        # Convert the tuples back to lists
        sorted_co_list = list(sorted_co_list)
        sorted_question_count_list = list(sorted_question_count_list)

        graph2[name_graph[1]][x[1]] = sorted_co_list
        graph2[name_graph[1]][y[1]] = sorted_question_count_list
        
        #GRAPH3

        graph3 = form(name_graph[2], x[2], y[2])
        for item,count in co_all_counts.items():
            # Kiểm tra xem phần tử có chứa cụm từ 'CO1' không
            for co in data_lo[highest_lo]:
                if co in item:
                    pattern = r'\b' + re.escape(item) + r'\b'
                    specific_chapter = self.df[self.df['subchapters_change'].str.contains(pattern, regex=True)]

                    question_difficulty_count = specific_chapter['difficulty'].value_counts()
                    dict0 = question_difficulty_count.to_dict()
                    dict1 ={}
                    for i in range(1, 4):
                        if i in list(question_difficulty_count.keys()):
                            dict1[i]= dict0[i]
                        else:
                            dict1[i] = 0
                        
                    graph3[name_graph[2]][x[2]].append(item)
                    graph3[name_graph[2]][y[2]].append(list(dict1.values()))

        co_list = graph3[name_graph[2]][x[2]]
        question_count_list = graph3[name_graph[2]][y[2]]

        # Function to convert "co" strings into tuples of integers for sorting
        def co_key(co_string):
            return tuple(map(int, co_string[2:].split('.')))

        # Combine the lists into pairs, sort by the co list using the custom key function
        sorted_pairs = sorted(zip(co_list, question_count_list), key=lambda pair: co_key(pair[0]))

        # Unzip the pairs back into two lists
        sorted_co_list, sorted_question_count_list = zip(*sorted_pairs)

        # Convert the tuples back to lists
        sorted_co_list = list(sorted_co_list)
        sorted_question_count_list = list(sorted_question_count_list)


        graph3[name_graph[2]][x[2]] = sorted_co_list
        graph3[name_graph[2]][y[2]] = sorted_question_count_list

        #GRAPH4
        graph4 = form(name_graph[3], x[3], y[3])

        for i in range(0, 100, 10):
            graph4[name_graph[3]][x[3]].append(str(i)+'%')

        graph4[name_graph[3]][y[3]] = [0]*10

        list_iou = []
        for i in self.df['spatial_matrix']:
            listMatrix = json.loads(i)
            matrix2 = np.array(listMatrix)
            list_iou.append(self.calculate_iou(matrix, matrix2))

        for i in list_iou:
            for t in range(90,-10,-10):
                if i >= t/100:
                    graph4[name_graph[3]][y[3]][int(t/10)] +=1
                    break
                

        return graph1|graph2|graph3|graph4






