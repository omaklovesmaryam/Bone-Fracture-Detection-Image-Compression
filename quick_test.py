# quick_test.py
import os
import pandas as pd
import matplotlib.pyplot as plt

def analyze_results():
    """Quick analysis of experiment results"""
    THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
    
    # This would be populated from your actual results
    compression_types = ['jpeg_90', 'jpeg_50', 'jpeg2000_20', 'jpeg2000_80']
    
    # Placeholder - you'll replace this with actual results
    results = {
        'jpeg_90': 0.85,
        'jpeg_50': 0.82,
        'jpeg2000_20': 0.84,
        'jpeg2000_80': 0.78
    }
    
    # Create comparison plot
    plt.figure(figsize=(10, 6))
    plt.bar(results.keys(), results.values())
    plt.title('Model Performance by Compression Type')
    plt.ylabel('Accuracy')
    plt.xlabel('Compression Type')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(THIS_FOLDER + '/plots/FractureDetection/Elbow/compression_comparison.png')
    plt.show()
    
    print("Compression Analysis Complete!")

if __name__ == "__main__":
    analyze_results()