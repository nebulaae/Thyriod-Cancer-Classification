import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np

def split_data(metadata_path='metadata.csv', n_splits=5, seed=42):
    df = pd.read_csv(metadata_path)
    
    # Stratified split based on label
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    df['fold'] = -1
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        df.loc[val_idx, 'fold'] = fold
        
    # Verify splits
    print("Split Verification:")
    for fold in range(n_splits):
        fold_df = df[df['fold'] == fold]
        counts = fold_df['label'].value_counts().to_dict()
        print(f"Fold {fold}: {len(fold_df)} patients, Label counts: {counts}")
        
    df.to_csv(metadata_path, index=False)
    print(f"\nUpdated {metadata_path} with fold assignments.")

if __name__ == "__main__":
    split_data()
