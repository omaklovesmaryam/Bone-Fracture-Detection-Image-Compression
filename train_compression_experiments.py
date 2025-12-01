# train_compression_metrics.py
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import time
import psutil
import json
from datetime import datetime

def get_model_complexity(model):
    """Calculate model complexity metrics"""
    trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = np.sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    
    return {
        'trainable_params': int(trainable_params),
        'non_trainable_params': int(non_trainable_params),
        'total_params': int(total_params),
        'model_layers': len(model.layers)
    }

def get_memory_usage():
    """Get current memory usage"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

def get_training_metrics(history, training_time, model, compression_config):
    """Extract comprehensive training metrics"""
    
    # Training time metrics
    time_metrics = {
        'total_training_time_seconds': training_time,
        'total_training_time_minutes': training_time / 60,
        'time_per_epoch_seconds': training_time / len(history.history['accuracy']),
        'completion_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Accuracy metrics
    final_train_accuracy = history.history['accuracy'][-1]
    final_val_accuracy = history.history['val_accuracy'][-1]
    
    # Best accuracy metrics
    best_val_accuracy = max(history.history['val_accuracy'])
    best_val_epoch = history.history['val_accuracy'].index(best_val_accuracy) + 1
    
    # Loss metrics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    # Convergence speed (epochs to reach 80% of max accuracy)
    target_accuracy = 0.8 * best_val_accuracy
    convergence_epoch = None
    for epoch, acc in enumerate(history.history['val_accuracy']):
        if acc >= target_accuracy:
            convergence_epoch = epoch + 1
            break
    
    accuracy_metrics = {
        'final_train_accuracy': final_train_accuracy,
        'final_val_accuracy': final_val_accuracy,
        'best_val_accuracy': best_val_accuracy,
        'best_val_epoch': best_val_epoch,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'convergence_epoch': convergence_epoch,
        'total_epochs_trained': len(history.history['accuracy'])
    }
    
    # Model complexity
    complexity_metrics = get_model_complexity(model)
    
    # Memory usage
    memory_metrics = {
        'peak_memory_usage_mb': get_memory_usage()
    }
    
    # Compression type info
    compression_info = {
        'compression_type': compression_config,
        'algorithm': 'JPEG' if 'jpeg' in compression_config else 'JPEG2000',
        'quality_level': compression_config.split('_')[-1]
    }
    
    # Combine all metrics
    all_metrics = {
        **compression_info,
        **time_metrics,
        **accuracy_metrics,
        **complexity_metrics,
        **memory_metrics
    }
    
    return all_metrics

def load_compressed_dataset(compression_config, part="Elbow"):
    """Load dataset for specific compression configuration"""
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    base_path = THIS_FOLDER + f'/Compressed_Elbow_Dataset/{compression_config}/'
    
    dataset = []
    
    for split in ['train', 'test']:
        split_path = os.path.join(base_path, split, part)
        if not os.path.exists(split_path):
            continue
            
        for img_file in os.listdir(split_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jp2')):
                img_path = os.path.join(split_path, img_file)
                
                # Extract label from filename
                if 'positive' in img_file or 'fractured' in img_file:
                    label = 'fractured'
                elif 'negative' in img_file or 'normal' in img_file:
                    label = 'normal'
                else:
                    continue
                
                dataset.append({
                    'label': label,
                    'image_path': img_path,
                    'compression': compression_config
                })
    
    return dataset

def train_compression_model_with_metrics(compression_config):
    """
    Train model and capture comprehensive metrics
    """
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    
    print(f"\n🚀 STARTING TRAINING: {compression_config}")
    print("=" * 50)
    
    start_time = time.time()
    
    # Load dataset
    data = load_compressed_dataset(compression_config, "Elbow")
    
    if not data:
        print(f"❌ No data found for: {compression_config}")
        return None
    
    print(f"📊 Loaded {len(data)} images for {compression_config}")
    
    labels = []
    filepaths = []
    
    for row in data:
        labels.append(row['label'])
        filepaths.append(row['image_path'])
    
    filepaths = pd.Series(filepaths, name='Filepath').astype(str)
    labels = pd.Series(labels, name='Label')
    
    images = pd.concat([filepaths, labels], axis=1)
    
    # Split dataset
    train_df, test_df = train_test_split(images, train_size=0.85, shuffle=True, random_state=1)
    
    # Data generators
    train_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input,
        validation_split=0.15
    )
    
    test_generator = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.resnet50.preprocess_input
    )
    
    train_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=True,
        seed=42,
        subset='training'
    )
    
    val_images = train_generator.flow_from_dataframe(
        dataframe=train_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=True,
        seed=42,
        subset='validation'
    )
    
    test_images = test_generator.flow_from_dataframe(
        dataframe=test_df,
        x_col='Filepath',
        y_col='Label',
        target_size=(224, 224),
        color_mode='rgb',
        class_mode='categorical',
        batch_size=16,
        shuffle=False
    )
    
    # Model architecture
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
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]
    
    # Training
    print(f"🎯 Training {compression_config} model...")
    history = model.fit(
        train_images,
        validation_data=val_images,
        epochs=12,
        callbacks=callbacks,
        verbose=1
    )
    
    training_time = time.time() - start_time
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(test_images, verbose=0)
    
    # Save model
    model_save_path = THIS_FOLDER + f"/weights/ResNet50_Elbow_{compression_config}_METRICS.h5"
    model.save(model_save_path)
    
    # Capture comprehensive metrics
    metrics = get_training_metrics(history, training_time, model, compression_config)
    metrics['test_accuracy'] = test_accuracy
    metrics['test_loss'] = test_loss
    
    # Print summary
    print(f"\n {compression_config} COMPLETED")
    print(f" Training Time: {metrics['total_training_time_minutes']:.1f} minutes")
    print(f"Final Test Accuracy: {metrics['test_accuracy']*100:.2f}%")
    print(f"Model Parameters: {metrics['total_params']:,}")
    print(f"Convergence Speed: {metrics['convergence_epoch']} epochs to 80% of max")
    
    # Create detailed plots
    create_detailed_plots(history, metrics, THIS_FOLDER)
    
    return metrics

def create_detailed_plots(history, metrics, base_folder):
    """Create comprehensive visualization plots"""
    
    # Main training plot
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history.history['accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(history.history['val_accuracy'], 'r-', label='Val Accuracy', linewidth=2)
    plt.axhline(y=metrics['best_val_accuracy'], color='g', linestyle='--', 
                label=f'Best Val: {metrics["best_val_accuracy"]:.3f}')
    plt.title(f'Accuracy - {metrics["compression_type"]}\nTest: {metrics["test_accuracy"]:.3f}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.plot(history.history['loss'], 'b-', label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    plt.title(f'Loss - {metrics["compression_type"]}\nTime: {metrics["total_training_time_minutes"]:.1f} min')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    # Convergence speed visualization
    epochs = range(1, len(history.history['val_accuracy']) + 1)
    plt.plot(epochs, history.history['val_accuracy'], 'g-', linewidth=2)
    if metrics['convergence_epoch']:
        plt.axvline(x=metrics['convergence_epoch'], color='orange', linestyle='--',
                   label=f'80% Convergence: Epoch {metrics["convergence_epoch"]}')
    plt.title(f'Convergence - {metrics["compression_type"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(base_folder, f"plots/FractureDetection/Elbow/{metrics['compression_type']}_DETAILED.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_comprehensive_experiments():
    """Run all experiments with full metric tracking"""
    compression_configs = ['jpeg_90', 'jpeg_50', 'jpeg2000_20', 'jpeg2000_80']
    
    print("STRTING COMPREHENSIVE COMPRESSION EXPERIMENTS")
    print("=" * 70)
    
    all_metrics = []
    total_start = time.time()
    
    for i, config in enumerate(compression_configs, 1):
        print(f"\nExperiment {i}/{len(compression_configs)}: {config}")
        print("-" * 50)
        
        try:
            metrics = train_compression_model_with_metrics(config)
            if metrics:
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Failed for {config}: {e}")
    
    total_time = time.time() - total_start
    
    # Save all metrics to JSON and CSV
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    
    with open('compression_experiment_metrics.json', 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    metrics_df = pd.DataFrame(all_metrics)
    try:
        existing_df = pd.read_csv('compression_experiment_metrics_complete.csv')
    # Only save new experiments if they don't exist
        new_experiments = metrics_df[~metrics_df['compression_type'].isin(existing_df['compression_type'])]
        if not new_experiments.empty:
            combined_df = pd.concat([existing_df, new_experiments], ignore_index=True)
            combined_df.to_csv('compression_experiment_metrics_complete.csv', index=False)
            print(f"Appended new results to complete file: {len(new_experiments)} new models")
        else:
            metrics_df.to_csv('compression_experiment_metrics.csv', index=False)
            print("Results saved to compression_experiment_metrics.csv")
    except:
        metrics_df.to_csv('compression_experiment_metrics.csv', index=False)
        print("Results saved to compression_experiment_metrics.csv")
    
    # Generate comprehensive analysis
    generate_final_analysis(metrics_df, total_time)
    
    return metrics_df

def generate_final_analysis(metrics_df, total_time):
    """Generate final analysis with all metrics"""
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EXPERIMENT ANALYSIS")
    print("=" * 80)
    
    print(f"Total Experiment Time: {total_time/60:.1f} minutes")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\nPERFORMANCE METRICS SUMMARY:")
    print("-" * 60)
    print(f"{'Compression':<15} {'Test Acc':<10} {'Train Time':<12} {'Params':<12} {'Convergence':<12}")
    print("-" * 60)
    
    for _, row in metrics_df.iterrows():
        print(f"{row['compression_type']:<15} {row['test_accuracy']*100:<9.1f}% {row['total_training_time_minutes']:<11.1f}m {row['total_params']/1e6:<11.1f}M {row['convergence_epoch']:<11}")
    
    # Create comprehensive comparison chart
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Accuracy Comparison
    plt.subplot(2, 3, 1)
    colors = ['#2E8B57', '#FFA500', '#1E90FF', '#FF6347']
    bars = plt.bar(metrics_df['compression_type'], metrics_df['test_accuracy'], color=colors, alpha=0.7)
    plt.title('Test Accuracy by Compression Type', fontweight='bold')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    for bar, acc in zip(bars, metrics_df['test_accuracy']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc*100:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Training Time Comparison
    plt.subplot(2, 3, 2)
    bars = plt.bar(metrics_df['compression_type'], metrics_df['total_training_time_minutes'], color=colors, alpha=0.7)
    plt.title('Training Time Comparison', fontweight='bold')
    plt.ylabel('Time (minutes)')
    plt.xticks(rotation=45)
    for bar, time_val in zip(bars, metrics_df['total_training_time_minutes']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{time_val:.1f}m', ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Convergence Speed
    plt.subplot(2, 3, 3)
    bars = plt.bar(metrics_df['compression_type'], metrics_df['convergence_epoch'], color=colors, alpha=0.7)
    plt.title('Convergence Speed (Epochs to 80% of max)', fontweight='bold')
    plt.ylabel('Epochs')
    plt.xticks(rotation=45)
    for bar, epoch in zip(bars, metrics_df['convergence_epoch']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{epoch}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Memory Usage
    plt.subplot(2, 3, 4)
    bars = plt.bar(metrics_df['compression_type'], metrics_df['peak_memory_usage_mb'], color=colors, alpha=0.7)
    plt.title('Peak Memory Usage', fontweight='bold')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    for bar, memory in zip(bars, metrics_df['peak_memory_usage_mb']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{memory:.0f}MB', ha='center', va='bottom', fontweight='bold')
    
    # Plot 5: Model Size (Parameters)
    plt.subplot(2, 3, 5)
    bars = plt.bar(metrics_df['compression_type'], metrics_df['total_params']/1e6, color=colors, alpha=0.7)
    plt.title('Model Complexity (Parameters)', fontweight='bold')
    plt.ylabel('Millions of Parameters')
    plt.xticks(rotation=45)
    for bar, params in zip(bars, metrics_df['total_params']/1e6):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{params:.1f}M', ha='center', va='bottom', fontweight='bold')
    
    # Plot 6: Efficiency Score (Accuracy per minute)
    efficiency = metrics_df['test_accuracy'] / metrics_df['total_training_time_minutes']
    plt.subplot(2, 3, 6)
    bars = plt.bar(metrics_df['compression_type'], efficiency, color=colors, alpha=0.7)
    plt.title('Training Efficiency\n(Accuracy per Minute)', fontweight='bold')
    plt.ylabel('Accuracy / Minute')
    plt.xticks(rotation=45)
    for bar, eff in zip(bars, efficiency):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{eff:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_compression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nresults saved to:")
    print("   - compression_experiment_metrics.json")
    print("   - compression_experiment_metrics.csv")
    print("   - comprehensive_compression_analysis.png")

if __name__ == "__main__":
    # Create directories
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(THIS_FOLDER + "/plots/FractureDetection/Elbow/", exist_ok=True)
    os.makedirs(THIS_FOLDER + "/weights/", exist_ok=True)
    
    # Run comprehensive experiments
    results_df = run_comprehensive_experiments()