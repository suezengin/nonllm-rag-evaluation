import pandas as pd
import ast
import os

prefix = ("../../data/inputs/")
reis_response_df = pd.read_excel (prefix + "reis_questions_responses.xlsx")
inboedel_response_df = pd.read_excel (prefix + "inboedel_questions_responses.xlsx")

def parse_full_response(response_str):
    try:
        data = ast.literal_eval(response_str)
        flat = {}
        for top_level_key in ['aegon_response', 'asr_response', 'comparision_response', 'additional_output']:
            content = data.get(top_level_key, {})
            if top_level_key == 'additional_output':
                sub_content = content.get('classification_check', {})
                for key, val in sub_content.items():
                    flat[f'{top_level_key}_classification_check_{key}'] = val
            else:
                for key, val in content.items():
                    flat[f'{top_level_key}_{key}'] = val
        return pd.Series(flat)   
    except Exception as e:
        print(f"Parse error: {e}")  
        return pd.Series()       

parsed_reis_responses = reis_response_df['response'].apply(parse_full_response)     
parsed_inboedel_responses = inboedel_response_df['response'].apply(parse_full_response)    

reis_cleaned = pd.concat([reis_response_df.drop(columns=['response']), parsed_reis_responses], axis=1)
inboedel_cleaned = pd.concat([inboedel_response_df.drop(columns=['response']), parsed_inboedel_responses], axis=1)

output_prefix = ("../../data/intermediate/")
reis_cleaned.to_excel(output_prefix + "parsed_reis_responses.xlsx", index= False, engine="openpyxl")
inboedel_cleaned.to_excel(output_prefix + "parsed_inboedel_responses.xlsx", index= False, engine="openpyxl")

print("Done, flies saved")