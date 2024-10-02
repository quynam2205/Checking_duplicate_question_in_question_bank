from google.generativeai import caching
import google.generativeai as genai
import datetime
import os
import PyPDF2
from python_algo.config import GOOGLE_API_KEY, generation_config, safety_settings, cache_time

global book_path
book_path = r"current\input_file\pythonlearn.pdf"

def read_pdf(file_path: str):
    pdf_data = ""  
    with open(file_path, "rb") as f:
        pdf = PyPDF2.PdfReader(f)
        for page in pdf.pages:
            new_page = page.extract_text()
            pdf_data = pdf_data + new_page
    return pdf_data

class Prompt():
    def prompt_explanation(self, 
                           question, 
                           context
                           ):
        out_format = """
        {
            "Id": "...",
            "Correct answer explanation": "..."
        }
        """
        prompt = f"""
        You are a code assistant, a highly advanced large language model have in-depth knowledge
        of Python programming. Your core strengths lie in tackling complex Python questions,
        utilizing intricate reasoning, and delivering solutions through methodical problem-solving.

        'python for everyone', which is a textbook about Python programming language. This
        textbook provides an Informatics-oriented introduction to programming. You responds have
        to based on the knowledge and context contained in this notebook.
        You responds have to based on the knowledge and context contained in below part.
        -------- 
        {context}
        --------
        Now, remember your primary objective is to dissect and address each problem with a rigorous
        and detailed approach. This task involves:
        1. Clearly identifying and understanding the question.
        2. Breaking down the question into question part and selection part.
        3. Focus on question part, find out all the python syntax or python code.
        Applying relevant Python principles and techniques to solve code line-by-line,
        explain the concept contained in each line. Do not try to explain code in selection part.
        Instead, remember the number of line of explanation must equal to the number of line of
        provided code.
        4. Understand the requirement and synthesizing the line-by-line to formulate a
        comprehensive answer to explain the correct answer.
        Integrate step-by-step reasoning to solve Python problems under following structure: {out_format}
        ###
        Here is description about each attribute:
        "Id": Index of the question
        "Correct answer explanation": Explain based on the correct answer. Do not explain anything about incorrect answer.
        Access to given file "Unit3.csv" to apply your task to all the question in the file.
        Begin your task from the question at the beginning and try to go through all the question.
        ###
        Question: {question['question_content']}
        Answer: {question['ans']}
        Id: {question['id']}
        """
        
        return prompt

    def prompt_check_dup(self, question1, question2):
        response_format = """
        {
            'Id': {
                Level:...,
                Reason:...
            },...
        }
        """
        prompt = f"""
        You are an expert in Python and a duplication-checking agent, you have an in-depth knowledge
        of Python programming. Your core strengths lie in tackling complex Python questions,
        utilizing intricate reasoning, and delivering solutions through methodical problem-solving.
        Throughout this interaction, you will encounter a variety of Python problems,
        ranging from basic theories to advanced algorithms.
        ###
        Your primary objective is to dissect and address each problem with a rigorous and detailed
        approach. This involves:
        1. Clearly identifying and understanding the problem statement.
        2. Breaking down the problem into manageable components to understand the topic, concept, 
        example, and context in question.
        3. Analyzing the correct answer and instruction of the question to understand how to solve 
        the problem and the step-by-step to the correct answer.
        4. Compare the given question with the original question to find out whether they are duplicates or 
        not. Analyze the grammatical structure and meaning of two questions. Determine whether they 
        have the same subject, main verb, and semantic object. Determine whether two questions refer
        to the same concept or entity.
        ###
        I will give you an original question and a list of questions. You have to comply with the above 
        thought process to compare each question in the list with the original question. Each 
        question will be evaluated through the following fields:
            'Question': Orginal question,
            'Correct answer': Correct answer of the question,
            'Instruction': Correct answer explanation of the question,
        ###
        Your response must follow format: {response_format}
        With 'Id' is the Id of the question to be compared with the original question , 'Level' is 0 for not duplicate or 1 for duplicate based on involves 4.
        Reason is the reason why you conclude that level.
        ###
        Original question: {question1}
        List of question: {question2}
        """

        return prompt
    
    def prompt_check_dup_all(self,original_question, list_question):

        prompt=f"""
            You are an expert in ranking the question. Your task is to rank the given list of 
            questions sequentially based on similarity to an original question.Your decision must based on aggregation
            of topic, concept, problem, logic of the questions.
            ###
            "Original question":  
                Question: {original_question['question_content']}
                Answer: {original_question['ans']}
                Id: {original_question['id']}
            ###
            "List of questions to be compared":
            [{list_question}]
            ###
            Your ranking decision will comply with the following JSON structure:
            {{
                "Rank 1": {{"id": , "reason": ,"similar_percent": }},
                "Rank 2": {{"id": , "reason": ,"similar_percent":}},
                ...
                "Rank n": {{"id": , "reason": ,"similar_percent":}}
            }}
            """
        # print('feddddddddddd')
        # print(prompt)
        
        return prompt
    


class LLM:
    def __init__(self):
        book = read_pdf(book_path)
        os.environ['GENAI_API_KEY'] = GOOGLE_API_KEY
        genai.configure(api_key=os.environ['GENAI_API_KEY'])

        self.prompt = Prompt()

        self.cache = caching.CachedContent.create(
            model="models/gemini-1.5-pro-001",
            display_name="python for everyone", 
            system_instruction="You are an expert in Python and a duplication checking agent, you have an in-depth knowledge of Python programming. Your core strengths lie in tackling complex Python questions, utilizing intricate reasoning, and delivering solutions through methodical problem-solving. Throughout this interaction, you will encounter a variety of Python problems, ranging from basic theories to advanced algorithms.",
            contents=[book],
            ttl=datetime.timedelta(minutes=cache_time),
        )
        self.LLM = genai.GenerativeModel.from_cached_content(cached_content=self.cache, generation_config=generation_config, safety_settings=safety_settings)


    def get_prompt(self, 
                   task_num, 
                   question_1, 
                   question_2, 
                   context = None
                   ):
      
        if task_num == 1:
            return  self.prompt.prompt_explanation(question_1, context)
        elif task_num == -1:
            return  self.prompt.prompt_check_dup_all(question_1, question_2)
        else:
            return  self.prompt.prompt_check_dup(question_1,question_2)

    def get_completion(self, 
                       prompt
                       ):
        result = self.LLM.generate_content(prompt)
        return result.text