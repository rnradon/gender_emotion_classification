import pandas as pd
from sklearn.model_selection import train_test_split

# ----This code segment will be shifted to the file where we need to import our data----"
# IMDB_DATASET_PATH = '../data/imdb_crop/'
# file = IMDB_DATASET_PATH + 'gender_df.csv'
# ------------------

def import_data(file_path, dataset_name):
    data = pd.read_csv(file_path, sep=',')
    
    if str(dataset_name) == "imdb":
        x_train, x_test, y_train, y_test = train_test_split(data['grayscale_array'], data['gender'], test_size=0.2, random_state=42, shuffle=False)
    
    return x_train, x_test, y_train, y_test