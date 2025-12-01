# train_jp2k.py
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time
try:
    import glymur
    GLYMUR_AVAILABLE = True
except ImportError:
    print("glymur not installed. Run: pip install glymur")
    GLYMUR_AVAILABLE = False

def jp2k_loader(file_path):
    """Load JPEG2000 files using glymur library - UPDATED FOR GRAYSCALE"""
    try:
        jp2_obj = glymur.Jp2k(file_path)
        img_array = jp2_obj[:]
        
        # X-RAY IMAGES ARE GRAYSCALE - Handle grayscale properly
        if len(img_array.shape) == 2:  # Grayscale - keep as is
            # Resize to 224x224 if needed
            if img_array.shape[:2] != (224, 224):
                img_array = tf.image.resize(img_array[..., np.newaxis], [224, 224])
                img_array = img_array.numpy() if hasattr(img_array, 'numpy') else img_array
                img_array = img_array[:, :, 0]  # Remove channel dimension
            return img_array
        else:
            # If somehow RGB, convert to grayscale
            img_gray = np.mean(img_array, axis=2).astype(np.uint8)
            if img_gray.shape[:2] != (224, 224):
                img_gray = tf.image.resize(img_gray[..., np.newaxis], [224, 224])
                img_gray = img_gray.numpy() if hasattr(img_gray, 'numpy') else img_gray
                img_gray = img_gray[:, :, 0]
            return img_gray
            
    except Exception as e:
        print(f"Error loading JPEG2000 {file_path}: {e}")
        return None

def preprocess_grayscale_image(img_array):
    """Preprocess grayscale image for ResNet50"""
    # Convert grayscale to RGB by repeating channels (ResNet50 expects 3 channels)
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    
    # Preprocess for ResNet50
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

def train_jp2k_models():
    """Fixed version that properly loads training data - UPDATED FOR GRAYSCALE"""
    if not GLYMUR_AVAILABLE:
        print("Please install glymur: pip install glymur")
        return
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    
    jp2k_configs = ['jpeg2000_20', 'jpeg2000_80']
    all_metrics = []
    
    for config in jp2k_configs:
        print(f"\nTRAINING JPEG2000: {config}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Load ALL images and manually split into train/test
        base_path = THIS_FOLDER + f'/Compressed_Elbow_Dataset/{config}/'
        all_images = []
        
        # Get all JPEG2000 files regardless of split
        for root, dirs, files in os.walk(os.path.join(base_path)):
            for file in files:
                if file.endswith('.jp2'):
                    full_path = os.path.join(root, file)
                    
                    # Extract label from filename
                    if 'positive' in file or 'fractured' in file:
                        label = 'fractured'
                    elif 'negative' in file or 'normal' in file:
                        label = 'normal'
                    else:
                        continue
                    
                    all_images.append({
                        'filepath': full_path,
                        'label': label
                    })
        
        if not all_images:
            print(f"No JPEG2000 files found for {config}")
            continue
            
        print(f"Found {len(all_images)} total JPEG2000 images")
        
        # Convert to DataFrame and manually split (80% train, 20% test)
        df = pd.DataFrame(all_images)
        
        # Manual train/test split
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        
        print(f"Training: {len(train_df)} images")
        print(f" Testing: {len(test_df)} images")
        
        if len(train_df) == 0:
            print(f"No training data after split for {config}")
            continue
        
        # Create data generators - UPDATED FOR GRAYSCALE
        def data_generator(dataframe, batch_size=16):
            """Generator for training data"""
            while True:
                # Shuffle each epoch
                shuffled_df = dataframe.sample(frac=1).reset_index(drop=True)
                
                for start_idx in range(0, len(shuffled_df), batch_size):
                    end_idx = min(start_idx + batch_size, len(shuffled_df))
                    batch_df = shuffled_df.iloc[start_idx:end_idx]
                    
                    batch_images = []
                    batch_labels = []
                    
                    for _, row in batch_df.iterrows():
                        img_array = jp2k_loader(row['filepath'])
                        if img_array is not None:
                            # Preprocess grayscale image for ResNet50
                            img_array = preprocess_grayscale_image(img_array)
                            batch_images.append(img_array)
                            
                            # Convert label to one-hot
                            label = 0 if row['label'] == 'fractured' else 1
                            batch_labels.append([1, 0] if label == 0 else [0, 1])
                    
                    if batch_images:
                        yield np.array(batch_images), np.array(batch_labels)
        
        # Build model
        pretrained_model = tf.keras.applications.ResNet50(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet',
            pooling='avg'
        )
        pretrained_model.trainable = False
        
        model = tf.keras.Sequential([
            pretrained_model,
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Calculate steps
        batch_size = 16
        train_steps = len(train_df) // batch_size
        test_steps = len(test_df) // batch_size
        
        # Train model
        print(f"Training {config} with {len(train_df)} images...")
        history = model.fit(
            data_generator(train_df, batch_size),
            steps_per_epoch=train_steps,
            epochs=12,
            validation_data=data_generator(test_df, batch_size),
            validation_steps=test_steps,
            verbose=1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            ]
        )
        
        training_time = time.time() - start_time
        
        # Final evaluation
        test_gen = data_generator(test_df, batch_size)
        test_loss, test_accuracy = model.evaluate(test_gen, steps=test_steps, verbose=0)
        
        # Save model
        model.save(f"{THIS_FOLDER}/weights/ResNet50_Elbow_{config}_GRAYSCALE.h5")
        
        # Collect metrics
        metrics = {
            'compression_type': config,
            'test_accuracy': test_accuracy,
            'training_time_minutes': training_time / 60,
            'total_params': model.count_params(),
            'algorithm': 'JPEG2000',
            'quality_level': config.split('_')[-1],
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'color_mode': 'grayscale'  # Add this to track
        }
        
        all_metrics.append(metrics)
        
        print(f"{config} COMPLETED")
        print(f"Test Accuracy: {test_accuracy*100:.2f}%")
        print(f" Training Time: {training_time/60:.1f} minutes")
        print(f" Train/Test split: {len(train_df)}/{len(test_df)} images")
    
    # Save results - UPDATED TO USE COMPLETE FILE
    if all_metrics:
        jp2k_df = pd.DataFrame(all_metrics)
        
        # Combine with existing JPEG results
        try:
            existing_df = pd.read_csv('compression_experiment_metrics_complete.csv')
            combined_df = pd.concat([existing_df, jp2k_df], ignore_index=True)
            combined_df.to_csv('compression_experiment_metrics_complete.csv', index=False)
            print(f"\n Combined results saved with {len(combined_df)} models!")
        except:
            # If complete file doesn't exist, try the regular one
            try:
                existing_df = pd.read_csv('compression_experiment_metrics.csv')
                combined_df = pd.concat([existing_df, jp2k_df], ignore_index=True)
                combined_df.to_csv('compression_experiment_metrics_complete.csv', index=False)
                print(f"\n Combined results saved with {len(combined_df)} models!")
            except:
                jp2k_df.to_csv('jp2k_metrics_grayscale.csv', index=False)
                print(" JPEG2000 results saved to jp2k_metrics_grayscale.csv")
    
    return all_metrics

if __name__ == "__main__":
    # First install glymur if needed
    if not GLYMUR_AVAILABLE:
        print("Installing glymur...")
        import subprocess
        subprocess.check_call(["pip", "install", "glymur"])
        import glymur
        print(" glymur installed successfully!")
    
    train_jp2k_models()