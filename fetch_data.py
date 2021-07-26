import pandas as pd
import os
import pickle
from sklearn.preprocessing import OneHotEncoder

root_path = "WESAD"
zip_label = "_E4_Data.zip"
E4_label = "_E4_Data"

pacient_state = {
    0: "Not Defined/Transient",
    1: "Baseline",
    2: "Stress",
    3: "Amusement",
    4: "Meditation",
    5: "Error/Not Labeled",
    6: "Error/Not Labeled",
    7: "Error/Not Labeled"
}

subjects_available = os.listdir(root_path)

def get_input_chest_data(subjects_list, body_signal):
    
    chest_df_list = []
    
    for subject in subjects_list:
        
        pkl_path = os.path.join(root_path, subject, subject + ".pkl")
        f=open(pkl_path,'rb')
        data=pickle.load(f,encoding='latin1')
        
        index_df = [i for i in range(1, len(data["label"]) + 1)]
        chest_data = {
            body_signal: data["signal"]["chest"][body_signal].reshape(len(data["signal"]["chest"][body_signal]),), 
            "subject": data["subject"],
            "label_id": data["label"],
            "label": [pacient_state[i]  for i in data["label"] ]
        }
        
        chest_df_list.append(pd.DataFrame(chest_data, index = index_df))
        
    return chest_df_list

input_chest_data = get_input_chest_data(subjects_available, "EDA")

if __name__ == "__main__": 

    print(len(input_chest_data))
    print(input_chest_data[0])