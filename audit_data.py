import os
import pandas as pd
import glob

def audit_dataset():
    # Define paths
    benign_us_dir = 'dataset2'
    benign_path_dir = 'dataset1'
    ptc_us_dir = 'dataset3'
    ptc_path_dir = 'dataset4'

    def get_patient_ids(directory, is_pathology=False):
        ids = set()
        for filename in os.listdir(directory):
            if is_pathology:
                # Expecting format XXX_YYY.tif
                patient_id = filename.split('_')[0]
            else:
                # Expecting format XXX.jpg or XXX.JPG
                patient_id = os.path.splitext(filename)[0]
            ids.add(patient_id)
        return ids

    benign_us_ids = get_patient_ids(benign_us_dir)
    ptc_us_ids = get_patient_ids(ptc_us_dir)
    benign_path_ids = get_patient_ids(benign_path_dir, is_pathology=True)
    ptc_path_ids = get_patient_ids(ptc_path_dir, is_pathology=True)

    print(f"Benign Ultrasound unique IDs: {len(benign_us_ids)}")
    print(f"PTC Ultrasound unique IDs: {len(ptc_us_ids)}")
    print(f"Benign Pathology unique IDs: {len(benign_path_ids)}")
    print(f"PTC Pathology unique IDs: {len(ptc_path_ids)}")

    # Check for ID collisions between Benign and PTC
    overlap_us = benign_us_ids.intersection(ptc_us_ids)
    print(f"ID overlap in US (Benign vs PTC): {overlap_us}")

    # Check for consistency within each class
    missing_path_benign = benign_us_ids - benign_path_ids
    missing_us_benign = benign_path_ids - benign_us_ids
    missing_path_ptc = ptc_us_ids - ptc_path_ids
    missing_us_ptc = ptc_path_ids - ptc_us_ids

    print(f"Benign patients missing Pathology: {missing_path_benign}")
    print(f"Benign patients missing Ultrasound: {missing_us_benign}")
    print(f"PTC patients missing Pathology: {missing_path_ptc}")
    print(f"PTC patients missing Ultrasound: {missing_us_ptc}")

    # Total unique patients across all sets
    all_patients = benign_us_ids | ptc_us_ids | benign_path_ids | ptc_path_ids
    print(f"Total unique patient IDs across all folders: {len(all_patients)}")

    # Create master metadata
    data = []

    # Benign (label 0)
    for pid in sorted(list(benign_us_ids), key=int):
        if pid in benign_path_ids:
            us_files = glob.glob(os.path.join(benign_us_dir, f"{pid}.*"))
            if not us_files: continue
            us_path = us_files[0]
            path_patches = sorted(glob.glob(os.path.join(benign_path_dir, f"{pid}_*.tif")))
            
            unique_id = f"benign_{pid}"
            data.append({
                'patient_id': unique_id,
                'original_id': pid,
                'label': 0,
                'ultrasound_image_path': us_path,
                'pathology_patch_paths': ";".join(path_patches),
                'num_patches': len(path_patches)
            })

    # PTC (label 1)
    for pid in sorted(list(ptc_us_ids), key=int):
        if pid in ptc_path_ids:
            us_files = glob.glob(os.path.join(ptc_us_dir, f"{pid}.*"))
            if not us_files: continue
            us_path = us_files[0]
            path_patches = sorted(glob.glob(os.path.join(ptc_path_dir, f"{pid}_*.tif")))
            
            unique_id = f"ptc_{pid}"
            data.append({
                'patient_id': unique_id,
                'original_id': pid,
                'label': 1,
                'ultrasound_image_path': us_path,
                'pathology_patch_paths': ";".join(path_patches),
                'num_patches': len(path_patches)
            })

    df = pd.DataFrame(data)
    
    # Validation
    print("\n--- Validation ---")
    print(f"Total paired patients: {len(df)}")
    print(df['label'].value_counts().rename({0: 'Benign', 1: 'PTC'}))
    
    patch_stats = df['num_patches'].value_counts().sort_index()
    print("\nPatch count distribution:")
    print(patch_stats)
    
    outliers = df[(df['num_patches'] < 3) | (df['num_patches'] > 7)]
    if not outliers.empty:
        print(f"\nWarning: {len(outliers)} patients have patch counts outside 3-7 range.")
        print(outliers[['patient_id', 'num_patches']])

    df.to_csv('metadata.csv', index=False)
    print("\nMetadata saved to metadata.csv")

if __name__ == "__main__":
    audit_dataset()
