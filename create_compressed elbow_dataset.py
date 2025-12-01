# create_compressed_elbow_dataset_fixed.py
import os
import cv2
import numpy as np
from PIL import Image
import pandas as pd

def create_compressed_elbow_dataset():
    """
    Create compressed versions of Elbow training images for 4 compression levels
    """
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    base_dir = THIS_FOLDER + '/Dataset/'
    output_base = THIS_FOLDER + '/Compressed_Elbow_Dataset/'
    
    # Compression configurations
    compression_configs = {
        'jpeg_90': {'type': 'jpeg', 'quality': 90},
        'jpeg_50': {'type': 'jpeg', 'quality': 50},
        'jpeg2000_20': {'type': 'jpeg2000', 'ratio': 20},
        'jpeg2000_80': {'type': 'jpeg2000', 'ratio': 80}
    }
    
    # Create output directories
    for config_name in compression_configs.keys():
        os.makedirs(os.path.join(output_base, config_name, 'train', 'Elbow'), exist_ok=True)
        os.makedirs(os.path.join(output_base, config_name, 'test', 'Elbow'), exist_ok=True)
    
    dataset_info = []
    
    # Process only Elbow images
    for split in ['train', 'test']:
        split_path = os.path.join(base_dir, split, 'Elbow')
        if not os.path.exists(split_path):
            continue
            
        for patient_folder in os.listdir(split_path):
            patient_path = os.path.join(split_path, patient_folder)
            if not os.path.isdir(patient_path):
                continue
                
            for label_folder in os.listdir(patient_path):
                if 'positive' in label_folder:
                    label = 'fractured'
                elif 'negative' in label_folder:
                    label = 'normal'
                else:
                    continue
                    
                label_path = os.path.join(patient_path, label_folder)
                
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(label_path, img_file)
                        
                        try:
                            # Load original image as grayscale
                            original_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if original_img is None:
                                continue
                                
                            # Resize to 224x224 for consistency
                            original_img = cv2.resize(original_img, (224, 224))
                            
                            # Create compressed versions
                            for config_name, config in compression_configs.items():
                                if config['type'] == 'jpeg':
                                    # JPEG compression
                                    output_path = os.path.join(
                                        output_base, config_name, split, 'Elbow', 
                                        f"{patient_folder}_{label_folder}_{img_file.split('.')[0]}.jpg"
                                    )
                                    
                                    # Save grayscale image with specified quality
                                    cv2.imwrite(output_path, original_img, 
                                               [cv2.IMWRITE_JPEG_QUALITY, config['quality']])
                                    
                                elif config['type'] == 'jpeg2000':
                                    # JPEG2000 compression using glymur (CORRECTED)
                                    output_path = os.path.join(
                                        output_base, config_name, split, 'Elbow',
                                        f"{patient_folder}_{label_folder}_{img_file.split('.')[0]}.jp2"
                                    )
                                    
                                    # Use glymur for proper JPEG2000 compression
                                    import glymur
                                    
                                    # Remove existing file if it exists
                                    if os.path.exists(output_path):
                                        os.remove(output_path)
                                    
                                    # Create JPEG2000 with specified compression ratio
                                    # cratios controls compression - lower numbers = more compression
                                    cratios = [config['ratio']]  # Single layer at target ratio
                                    
                                    # Use grayscale image directly (no BGR to RGB conversion needed)
                                    jp2 = glymur.Jp2k(output_path, data=original_img, cratios=cratios)
                                    
                                    print(f"Created JPEG2000 {config_name}: {output_path}")
                                
                                dataset_info.append({
                                    'config': config_name,
                                    'split': split,
                                    'patient_id': patient_folder,
                                    'label': label,
                                    'original_path': img_path,
                                    'compressed_path': output_path
                                })
                                
                        except Exception as e:
                            print(f"Error processing {img_path}: {e}")
                            continue
    
    # Save dataset info
    df = pd.DataFrame(dataset_info)
    df.to_csv(os.path.join(output_base, 'dataset_info.csv'), index=False)
    print(f"Created compressed dataset with {len(df)} images")
    return output_base

if __name__ == "__main__":
    # Check if glymur is installed
    try:
        import glymur
        print("✅ glymur is available")
        create_compressed_elbow_dataset()
    except ImportError:
        print("❌ glymur not installed. Installing...")
        import subprocess
        subprocess.check_call(["pip", "install", "glymur"])
        import glymur
        print("✅ glymur installed successfully!")
        create_compressed_elbow_dataset()