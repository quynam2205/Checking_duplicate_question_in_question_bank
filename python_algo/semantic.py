from python_algo.config import MODEL_SEMANTIC, Test_set_path
from sentence_transformers import util
import pandas as pd
import ast


def Semantic_calculate(question, 
                       bank_set
                       ):

    sentence_input = question['question_content']
    sentences_bank = bank_set
    input_embeddings = MODEL_SEMANTIC.encode(sentence_input, convert_to_tensor=True)
    bank_embeddings = MODEL_SEMANTIC.encode(sentences_bank, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(input_embeddings, bank_embeddings)
    return cosine_similarities

def Semantic_eval(predict_set):
    df = pd.read_csv(Test_set_path)
    sentences = df['instruction'].tolist()

    predict_embeddings = MODEL_SEMANTIC.encode(predict_set, convert_to_tensor=True)
    test_embeddings = MODEL_SEMANTIC.encode(sentences, convert_to_tensor=True)
    cosine_similarities = util.cos_sim(predict_embeddings, test_embeddings)

    num_questions = len(df)
    Sum = 0
    for i in range(num_questions):
        Sum+=cosine_similarities[i][i]

    return Sum/num_questions