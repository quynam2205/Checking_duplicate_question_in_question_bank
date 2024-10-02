import json
class statistic:
    def __init__(self, pdf_file=r'data\log_data\log_data2.json'):
        self.pdf_file = pdf_file
    def statistic_manual(self):
        data = []
        for i in self.pdf_file:
            with open(i, 'r') as json_file:
                data.append(json.load(json_file))
        statistic_data = {'request': 'ranking_question', 'number': []}
        statistic_data['number'].append(len(data[i]))

        dict = {}
        for i in data[1]:
            for t in i['question_input']['subchapters']:
                if t not in dict:
                    dict[t] = [0]*3
                for a in range(3):
                    if i['question_input']['difficulty'] is None:
                        i['question_input']['difficulty'] = 2
                    if int(i['question_input']['difficulty']) == a+1:
                        dict[t][a] +=1

        graph = {}
        graph['subchapter'] = [i for i in dict.keys()]
        graph['difficult_level'] = [i for i in dict.values()]

        result = {'general': statistic_data, 'rank_question': graph}
        return result
