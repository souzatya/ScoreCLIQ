import pandas as pd
from sklearn.model_selection import train_test_split

def set_input_format(df):
    pdf = pd.DataFrame()
    pdf['ItemNum'] = df['ItemNum'].astype(str)
    pdf['ItemStem'] = df['ItemStem_Text'].astype(str)
    
    option_cols = ['Answer__A', 'Answer__B', 'Answer__C', 'Answer__D', 'Answer__E', 'Answer__F', 'Answer__G', 'Answer__H', 'Answer__I', 'Answer__J']
    
    for idx, row in df.iterrows():
        for col in option_cols:
            if col in df.columns and pd.notna(row[col]):
                pdf.loc[idx, 'ItemStem'] += "\n" + str(row[col])
    
    pdf['Difficulty'] = df['Difficulty'].astype(str)
    
    return pdf

def train_val_split(train_df, val_size=0.3, seed=42):
    train, val = train_test_split(train_df, test_size=val_size, random_state=seed)
    return train, val