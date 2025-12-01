import pandas as pd
import matplotlib.pyplot as plt#for plots
import numpy as np#numerical ops
from compression import CompressionAnalyzer  # Your compression module
import os

class CompressionPerformanceAnalyzer:
    
    def __init__(self):
        self.results = []#here we keep all the results of findings
    
    def run_analysis(self, test_images, qualities=[90, 75, 50, 25], compression_ratios=[10, 20, 40, 80]):
    
        #here we run compression tests, on several imgs and then store findings
        analyzer = CompressionAnalyzer()
        #Much of this code is similar to tests in the original compression file
        for img_path in test_images:
            if os.path.exists(img_path): 
                print(f"now testing {img_path}")
                original_size = os.path.getsize(img_path)#og file size
        #here we test jpeg:
                for quality in qualities:
                    print(f"testing JPEG w/ quality at {quality}%...")
                    
                    #testing compression
                    compressed_img, ratio, psnr, mse = analyzer.test_single_image(img_path, standard='JPEG', quality=quality)
                    
                    #testing ai prediction, we get true or false based on if it was able to guess right after compression
                    prediction_changed = analyzer.test_prediction(img_path,  standard='JPEG', quality=quality)
                    # Calculate compressed file size
                    compressed_size = original_size / ratio
                    #store results/ans
                    self.results.append({
                        'image_path': img_path,
                        'standard': 'JPEG',
                        'quality': quality,
                        'compression_ratio': ratio,
                        #next 1 is important: it basically has T or F for each quality level, for each img
                        'prediction_changed': prediction_changed,
                        'file_size_reduction': (1 - ratio) * 100,  # % reduction
                        'psnr': psnr,
                        'mse': mse,
                        'compressed_size': compressed_size,  #acc file size in bytes
                        'original_size': original_size
                    })
        #here we test jpeg2000
                for compression_ratio in compression_ratios:
                    print(f"testing JPEG2000 w/ ratio {compression_ratio}:1")
                    
                    #testing compression
                    compressed_img, ratio, psnr, mse = analyzer.test_single_image(
                        img_path, standard='JPEG2000', compression_ratio=compression_ratio)
                    #testing ai prediction
                    prediction_changed = analyzer.test_prediction(
                        img_path, standard='JPEG2000', compression_ratio=compression_ratio)
                    
                    #storing ans
                    self.results.append({
                        'image_path': img_path,
                        'standard': 'JPEG2000',
                        'quality': compression_ratio,  # Using ratio as "quality" for plotting
                        'compression_ratio': ratio,
                        'prediction_changed': prediction_changed,
                        'file_size_reduction': (1 - ratio) * 100,
                        'psnr': psnr,
                        'mse': mse,
                        'compressed_size': compressed_size,  #acc file size in bytes
                        'original_size': original_size
                    })
                print("-" * 30)#just a seperation
    
    #THIS FUNCTION IS SO THAT THE LEGEND HAS IMAGE NAMES THAT MAKE SENSE.
    def img_labeler(self, img_path):
    #for ex. the image @ this path "Dataset/test/Elbow/patient11836/study1_positive/image2.png"
    #gets named: "Elbow_fractured"
        parts = img_path.split('/')
        body_part = parts[2]  #like "elbow", "hand", "shoulder"
        fracture_status = "fractured" if "positive" in img_path else "normal" #this is rlly clever basically it attaches if its fractureohfhosdhgs 
        return f"{body_part}_{fracture_status}"
    
    def create_performance_plots(self):
        #makes plots showing compression vs ai performance
        if not self.results:
            print("nothing to plot. must run tests first.")#ok now i know what is wrong this is debug
            return
        #first we organize data into a table in prep for plotting
        df = pd.DataFrame(self.results)
        
        #first plot is for Compression Ratio vs Quality
        #had to make the figure longer for quality metric plots, and now 8 for jpeg2000 plots
        plt.figure(figsize=(20, 8))#here is the plot window ty ai for teachingme
        #deepseek credit for helping me set up a plot
#JPEG PLOTS-------------------------------
    #1st plot compress ratio vs quality
        plt.subplot(2, 5, 1)#like matlab row,col,position
        jpeg_data = df[df['standard'] == 'JPEG']
        for img_path in jpeg_data['image_path'].unique():#go thru table
            img_data = jpeg_data[jpeg_data['image_path'] == img_path]#get data for an image
            label = self.img_labeler(img_path)#added this to give a better name to images
            plt.plot(img_data['quality'], img_data['compression_ratio'], 
                    marker='o', label=label)#plot
        plt.xlabel('Quality Level (%)')
        plt.ylabel('Compression Ratio')
        plt.title('JPEG: Quality vs Compression Ratio')
        plt.legend()
        plt.grid(True)
        plt.gca().invert_xaxis()
        #second si for File Size Reduction vs Quality
        plt.subplot(2, 5, 2)
        #similar method to plot one:
        for img_path in jpeg_data['image_path'].unique():
            img_data = jpeg_data[jpeg_data['image_path'] == img_path]
            label = self.img_labeler(img_path)
            plt.plot(img_data['quality'], img_data['file_size_reduction'], 
                    marker='s', label=label)
        plt.xlabel('Quality Level (%)')
        plt.ylabel('File Size Reduction (%)')
        plt.title('JPEG:Quality vs File Size Reduction')
        plt.grid(True)
        plt.gca().invert_xaxis()
        #plot 3= PSNR vs Quality
        plt.subplot(2, 5, 3) 
        for img_path in jpeg_data['image_path'].unique():
            img_data = jpeg_data[jpeg_data['image_path'] == img_path]
            plt.plot(img_data['quality'], img_data['psnr'], 
                    marker='o', label=os.path.basename(img_path))
        plt.xlabel('Quality Level (%)')
        plt.ylabel('PSNR (dB)')
        plt.title('Quality vs PSNR')
        plt.grid(True)
        plt.gca().invert_xaxis()
        #plot 4 MSE vs Quality  
        plt.subplot(2, 5, 4)
        for img_path in jpeg_data['image_path'].unique():
            img_data = jpeg_data[jpeg_data['image_path'] == img_path]
            plt.plot(img_data['quality'], img_data['mse'], 
                    marker='s', label=os.path.basename(img_path))
        plt.xlabel('Quality Level (%)')
        plt.ylabel('MSE')
        plt.title('Quality vs MSE')
        plt.grid(True)
        plt.gca().invert_xaxis()
        #5th plot: !!!this one is basically how prediction accuracy is affected by compression
        plt.subplot(2, 5, 5)

        #here we take the percent of changed answers, for each quality
        #btw this means its taking the percent changed for each qual. over ALL images at that quality
        jpeg_errors = jpeg_data.groupby('quality')['prediction_changed'].mean() * 100
        plt.bar(jpeg_errors.index, jpeg_errors.values, width=15)#i wanted thicker bars

        #setting x-axis to only show the qualities we tested
        plt.xticks(jpeg_errors.index)
        plt.xlabel('Quality Level (%)')
        plt.ylabel('Prediction Error Rate (%)')
        plt.title('Quality vs Models Prediction Errors')#idk how to use an appostraphy without breaking this
        plt.grid(True)
        plt.gca().invert_xaxis()
#JPEG2000 PLOTS-------------------------------

        # Plot 6: JPEG2000 Compression Ratio vs Compression Ratio
        plt.subplot(2, 5, 6)
        jpeg2k_data = df[df['standard'] == 'JPEG2000']
        for img_path in jpeg2k_data['image_path'].unique():
            img_data = jpeg2k_data[jpeg2k_data['image_path'] == img_path]
            label = self.img_labeler(img_path)
            plt.plot(img_data['quality'], img_data['compression_ratio'], 
                marker='o', label=label)
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('Compression Ratio')
        plt.title('JPEG2000: Ratio vs Compression')
        plt.legend(fontsize=8)
        plt.grid(True)
    
        # Plot 7: JPEG2000 File Size Reduction vs Compression Ratio
        plt.subplot(2, 5, 7)
        for img_path in jpeg2k_data['image_path'].unique():
            img_data = jpeg2k_data[jpeg2k_data['image_path'] == img_path]
            label = self.img_labeler(img_path)
            plt.plot(img_data['quality'], img_data['file_size_reduction'], 
                marker='s', label=label)
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('File Size Reduction (%)')
        plt.title('JPEG2000: Ratio vs File Size Reduction')
        plt.grid(True)
    
        # Plot 8: JPEG2000 PSNR vs Compression Ratio
        plt.subplot(2, 5, 8)
        for img_path in jpeg2k_data['image_path'].unique():
            img_data = jpeg2k_data[jpeg2k_data['image_path'] == img_path]
            plt.plot(img_data['quality'], img_data['psnr'], 
                marker='o', label=os.path.basename(img_path))
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('PSNR (dB)')
        plt.title('JPEG2000: Ratio vs PSNR')
        plt.grid(True)

    # Plot 9: JPEG2000 MSE vs Compression Ratio  
        plt.subplot(2, 5, 9)
        for img_path in jpeg2k_data['image_path'].unique():
            img_data = jpeg2k_data[jpeg2k_data['image_path'] == img_path]
            plt.plot(img_data['quality'], img_data['mse'], 
                marker='s', label=os.path.basename(img_path))
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('MSE')
        plt.title('JPEG2000: Ratio vs MSE')
        plt.grid(True)

    # Plot 10: JPEG2000 Prediction Errors
        plt.subplot(2, 5, 10)
        jpeg2k_errors = jpeg2k_data.groupby('quality')['prediction_changed'].mean() * 100
        plt.bar(jpeg2k_errors.index, jpeg2k_errors.values, width=15)
        plt.xticks(jpeg2k_errors.index)
        plt.xlabel('Compression Ratio (X:1)')
        plt.ylabel('Prediction Error Rate (%)')
        plt.title('JPEG2000: Ratio vs Prediction Errors')
        plt.grid(True)
        
        #deep seek taught me this
        plt.tight_layout()
        plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        self.print_summary(df)
    
    def print_summary(self, df):
        """print analysis summary"""
        print("\n" + "="*50)#adding visual divider
        print("COMPRESSION ANALYSIS SUMMARY")
        print("="*50)
        
        total_tests = len(df)
        errors = df['prediction_changed'].sum()
        error_rate = (errors / total_tests) * 100
        
        print(f"Total tests: {total_tests}")
        print(f"Prediction errors: {errors}")
        print(f"Overall error rate: {error_rate:.1f}%")
        
        print("\nError rates by quality level:")
        for quality in sorted(df['quality'].unique()):
            quality_data = df[df['quality'] == quality]
            quality_errors = quality_data['prediction_changed'].sum()
            quality_rate = (quality_errors / len(quality_data)) * 100
            avg_compression = quality_data['compression_ratio'].mean()
            print(f"  Quality {quality}%: {quality_rate:.1f}% errors, {avg_compression:.2f} compression ratio")

#idk
if __name__ == "__main__":
    #we can make it sift thru data later
    test_images = [
        "Dataset/test/Elbow/patient11836/study1_positive/image2.png",#subtle one (got it right)
        #"Dataset/test/Elbow/patient11818/study1_positive/image2.png",#very subtle & makes a mistake
        "Dataset/test/Elbow/patient11389/study1_positive/image2.png",#obvious one (got it right)
        "Dataset/test/Elbow/patient11204/study1_negative/image1.png",
        "Dataset/test/Hand/patient11190/study1_negative/image1.png",
        "Dataset/test/Hand/patient11194/study1_positive/image1.png",
        "Dataset/test/Shoulder/patient11186/study1_positive/image1.png",
        "Dataset/test/Shoulder/patient11187/study1_negative/image1.png"
    ]
    
    
    # here is where we run/call the analysis
    analyzer = CompressionPerformanceAnalyzer()
    analyzer.run_analysis(test_images)
    analyzer.create_performance_plots()