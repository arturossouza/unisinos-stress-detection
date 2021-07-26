from os import stat
from sklearn.preprocessing import OneHotEncoder
from scipy import stats

from fetch_data import input_chest_data, pacient_state

pacient_state_list  = [[state, pacient_state[state]] for state in pacient_state]

enc = OneHotEncoder(handle_unknown="ignore")
enc.fit(pacient_state_list)

clean_chest_data = []

for data in input_chest_data: 

    # appling One Hot Enconding in the labels

    aux_label = data.pop("label").to_numpy()
    aux_label_id = data.pop("label_id").to_numpy()

    aux = [[aux_label[i], aux_label_id[i]] for i in range(len(data))]

    target = enc.transform(aux).toarray()

    # normalizing data

    subject_label = data.pop("subject").to_numpy()[0]

    normalized_data = stats.zscore(data.to_numpy())

    # Tuple (input_data, target_data, subject_label)

    clean_chest_data.append((normalized_data, target, subject_label))


if __name__ == "__main__": 

    print(target)
    print(subject_label)
    print(normalized_data)
    print(clean_chest_data[0])
