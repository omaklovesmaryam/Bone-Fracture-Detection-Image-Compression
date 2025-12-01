# baseline_compression_tests_correct.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import sys


def test_with_original_predict():
    """Test compressed images using the original predict function"""
    
    # Try to import the original predict function
    try:
        # Import based on your original code structure
        from training_fractures import load_path  # or whatever module has predict
        print("Successfully imported original modules")
    except ImportError as e:
        print(f"Could not import original modules: {e}")
        print("Make sure you're in the same directory as your original project")
        return None
    
    # Since we can't directly import predict, let's create a compatible testing approach
    return test_compressed_images_direct()

def test_compressed_images_direct():
    """Test compressed images by mimicking the original prediction approach"""
    
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    
    # Load the original elbow model directly
    try:
        original_model_path = THIS_FOLDER + "/weights/ResNet50_Elbow_frac.h5"
        if not os.path.exists(original_model_path):
            print(f"Original model not found at: {original_model_path}")
            return None
            
        import tensorflow as tf
        model = tf.keras.models.load_model(original_model_path, compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print(" Original elbow model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    compression_types = ['jpeg_90', 'jpeg_50', 'jpeg2000_20', 'jpeg2000_80']
    baseline_results = {}
    
    print("\n" + "="*60)
    print("BASELINE TESTING WITH ORIGINAL PREDICT APPROACH")
    print("="*60)
    
    for compression_type in compression_types:
        print(f"\nTesting {compression_type}...")
        
        base_path = THIS_FOLDER + f'/Compressed_Elbow_Dataset/{compression_type}/test/Elbow/'
        if not os.path.exists(base_path):
            print(f"Path not found: {base_path}")
            continue
        
        # Collect all test images and their true labels
        test_images = []
        true_labels = []
        image_paths = []
        
        for img_file in os.listdir(base_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg', '.jp2')):
                img_path = os.path.join(base_path, img_file)
                
                # Extract true label from filename
                if 'positive' in img_file or 'fractured' in img_file:
                    true_label = 0  # fractured
                elif 'negative' in img_file or 'normal' in img_file:
                    true_label = 1  # normal
                else:
                    continue
                
                image_paths.append(img_path)
                true_labels.append(true_label)
        
        if not image_paths:
            print(f" No test images found for {compression_type}")
            continue
        
        print(f"Found {len(image_paths)} test images for {compression_type}")
        
        # Predict using the original model approach
        predictions = []
        correct_predictions = 0
        
        for i, img_path in enumerate(image_paths):
            try:
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
                img_array = np.expand_dims(img_array, axis=0)
                
                # Make prediction
                prediction = model.predict(img_array, verbose=0)
                predicted_class = np.argmax(prediction, axis=1)[0]
                predictions.append(predicted_class)
                
                # Check if correct
                if predicted_class == true_labels[i]:
                    correct_predictions += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        # Calculate accuracy
        if predictions:
            accuracy = correct_predictions / len(predictions)
            baseline_results[compression_type] = accuracy
            print(f" Original model accuracy on {compression_type}: {accuracy*100:.2f}%")
            
            # Show some prediction details
            print(f"   Correct: {correct_predictions}/{len(predictions)}")
    
    return baseline_results

def create_comparison_chart(comparison_data):
    """Create visualization comparing baseline vs fine-tuned performance"""
    compression_types = [data['compression_type'] for data in comparison_data]
    baseline_acc = [data['baseline_accuracy'] for data in comparison_data]
    fine_tuned_acc = [data['fine_tuned_accuracy'] for data in comparison_data]
    
    x = np.arange(len(compression_types))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, baseline_acc, width, label='Baseline (Original Model)', 
                   color='lightblue', alpha=0.7, edgecolor='black')
    bars2 = ax.bar(x + width/2, fine_tuned_acc, width, label='Fine-Tuned (Our Models)', 
                   color='lightgreen', alpha=0.7, edgecolor='black')
    
    ax.set_xlabel('Compression Type', fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontweight='bold')
    ax.set_title('Baseline vs Fine-Tuned Model Performance on Compressed Images', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(compression_types, rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('baseline_vs_finetuned_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def compare_with_our_results(baseline_results):
    """Compare baseline results with our fine-tuned models"""
    print("\n" + "="*60)
    print("BASELINE vs FINE-TUNED COMPARISON")
    print("="*60)
    
    # Load our fine-tuned results
    try:
        our_results_df = pd.read_csv('final_clean_results.csv')
        our_results = {}
        for _, row in our_results_df.iterrows():
            comp_type = row['compression_type'].replace('_corrected', '')
            our_results[comp_type] = row['test_accuracy']
        
        # Create comparison table
        comparison_data = []
        for comp_type in ['jpeg_90', 'jpeg_50', 'jpeg2000_20', 'jpeg2000_80']:
            if comp_type in baseline_results and comp_type in our_results:
                baseline_acc = baseline_results[comp_type] * 100
                our_acc = our_results[comp_type] * 100
                improvement = our_acc - baseline_acc
                
                comparison_data.append({
                    'compression_type': comp_type,
                    'baseline_accuracy': baseline_acc,
                    'fine_tuned_accuracy': our_acc,
                    'improvement': improvement
                })
        
        # Display comparison
        print(f"\n{'Compression':<15} {'Baseline':<12} {'Fine-Tuned':<12} {'Improvement':<12}")
        print("-" * 55)
        for data in comparison_data:
            print(f"{data['compression_type']:<15} {data['baseline_accuracy']:<11.1f}% {data['fine_tuned_accuracy']:<11.1f}% {data['improvement']:+.1f}%")
        
        # Create comparison visualization
        create_comparison_chart(comparison_data)
        
        # Save comparison results
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df.to_csv('baseline_vs_finetuned_comparison.csv', index=False)
        print(f"\n Comparison saved to 'baseline_vs_finetuned_comparison.csv'")
        
        # Generate insights
        generate_insights(comparison_data)
        
    except Exception as e:
        print(f"Could not load fine-tuned results: {e}")
        print(" Please ensure 'final_clean_results.csv' exists")

def generate_insights(comparison_data):
    """Generate insights from the comparison results"""
    print("\n KEY INSIGHTS:")
    print("-" * 25)
    
    total_improvement = 0
    count = 0
    
    for data in comparison_data:
        improvement = data['improvement']
        total_improvement += improvement
        count += 1
        
        if improvement > 0:
            print(f" {data['compression_type']}: +{improvement:.1f}% improvement with fine-tuning")
        else:
            print(f" {data['compression_type']}: {improvement:.1f}% (investigate)")
    
    if count > 0:
        avg_improvement = total_improvement / count
        print(f"\n Average improvement: +{avg_improvement:.1f}%")
        
        if avg_improvement > 0:
            print(" Fine-tuning on compressed data provides consistent performance gains")
        else:
            print(" Fine-tuning shows mixed results - compression may affect feature learning")

def main():
    print("BASELINE COMPRESSION PERFORMANCE ANALYSIS")
    print("Testing original elbow model on compressed images using direct prediction...")
    
    baseline_results = test_compressed_images_direct()
    
    if baseline_results:
        compare_with_our_results(baseline_results)
    else:
        print("Could not obtain baseline results")

if __name__ == "__main__":
    main()