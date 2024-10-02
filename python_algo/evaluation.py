# from sentence_transformers import SentenceTransformer
import pandas as pd
# import json
from python_algo.config import cache_cost_per_token, cache_cost_per_hour, cache_time, token_in, token_out
# import math

data_dup = pd.read_csv(r"data\question_dup_count.csv")


def cost_in_out(metadata: object) -> float:
    return (metadata.candidates_token_count * token_out) + ((metadata.prompt_token_count - metadata.cached_content_token_count)*token_in) + (metadata.cached_content_token_count * cache_cost_per_token)

def total_cost(metadata: object,
               in_out_cost: float) -> float:
    print(cache_cost_per_hour*cache_time*metadata.cached_content_token_count)
    return ((cache_cost_per_hour*cache_time*metadata.cached_content_token_count) + in_out_cost + (metadata.cached_content_token_count * token_in))