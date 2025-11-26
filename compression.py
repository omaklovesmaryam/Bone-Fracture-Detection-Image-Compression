import io #to use bytes that are in memory i think
import os #to see file paths, sizes idk
from PIL import Image #this is basically img compression stuff from the pillow library we use
import numpy as np #for operations


#here we handle our experiments with testing topologies
#topologies will be comprised up of a standard + a quality lvl
#we decided to start with jpeg in last meeting as a first step
class CompressionAnalyzer:

    def __init__(self):#constructor"-autoruns when we 
                       #create a new obj in this class. gives initial obj state
    
        #list of standards, each obj needs a copy of this:
        self.supported_standards = ['JPEG', 'JPEG2000']
        #print("initialized")

#this is a public intrface
    def compress_img(self, img_path,standard='JPEG',quality=90,compression_ratio=20):
    #self: ref to object itself,imgpath:file path to img to compress
    #standard: string name of standard. quality: int value
        print(f"found: {img_path}")
        original_img = Image.open(img_path)#load img frm disk
        if standard =='JPEG':
         return self._jpeg_compress(original_img, quality)#calls JPEG fxn
        elif standard == 'JPEG2000':
            return self._jpeg2000_compress(original_img, compression_ratio)
        else:
            raise ValueError(f"standard {standard} not implemented yet")
        
#Here is compression function for: JPEG
    def _jpeg_compress(self,img,quality):
    #PRIVATE-use only in this class.
        buffer = io.BytesIO()#creates virtual file in the RAM
    
    #Compression:
        img.save(buffer, format='JPEG', quality=quality)#frm Pillow; saves img to file with
       #automatic: 1.colourspace-conv 2.DCT xform, 3.Quantization, 4.Huffman Coding
        buffer.seek(0)#saving mean we are end of file, so here we go to start
        compressed_img = Image.open(buffer)#load compressed vers.
        print(f"JPEG compression successful.Quality:{quality}")
        return compressed_img
    
    def _jpeg2000_compress(self, img, compression_ratio=20):
    #this std uses wavelet xform + EBCOT Coding
    #compression_ratio: Target size reduction (e.g., 20 = 20:1 compression)  
    #convert PIL img to numpy array for processing
 #here is where we'll save

    #for JPEG2000, we specify compression ratio instead of quality in the params 
    #hi ratio = more compression!!

    #idea of glymur came from the source below:
    #https://stackoverflow.com/questions/29198255/python-jpeg2000-compression-with-glymur
        try:
            import glymur
            print(f"performing jpeg2000....")
        #conv PIL to numpy array SINCE: jpeg2000 is not supported in PIL :(
            img_array = np.array(img)
        #O/P img path, codestream o/p
            temp_output = "temp_compressed.jp2"
            #use glymur with compression ratios parameter

            if os.path.exists(temp_output):#if theres one already, delerte it
                os.remove(temp_output)
        # cratios controls the compression - lower numbers = more compression
        # Example: cratios=[192, 1] means first layer at 192:1, second at 1:1
            target_ratio = compression_ratio
       # Create compression ratios - we'll use two layers for better control
            cratios = [compression_ratio] # First layer at target ratio, second at 1:1 (highest quality)
        #using glymur to compress w/ JPEG2000 DOES WAVELET XFORM ITSELF!!!
        # Proper lossless/lossy encoding
            jp2 = glymur.Jp2k(temp_output, data=img_array, cratios=cratios)
        #glymur handles the wavelet transform and EBCOT coding internally
        
        #read back the compressed image
            compressed_array = jp2[:]  # Decode
            compressed_img = Image.fromarray(compressed_array)
        
            print(f"JPEG2000 compression complete! Ratio: {compression_ratio}:1")
            return compressed_img
        
        except ImportError:
            print("glymur not installed. Run: pip install glymur")
            raise
#Here is where where testing is "streamlined"? idk if thsts a good word
    def test_single_image(self, img_path, standard='JPEG', quality=75, compression_ratio=20):#quality=75 is defult value
        print(f"testing on {img_path}")
        original_size = os.path.getsize(img_path)#check OG file size
        print(f"Original size: {original_size} bytes")

        #compress it & save temporarily
        if standard == 'JPEG':
            compressed_img = self.compress_img(img_path, 'JPEG', quality)
            temp_path = "temp_compressed.jpg"
            temp_files_to_clean = [temp_path]#had to add cuz of jpeg2000
        else:  # JPEG2000
            compressed_img = self.compress_img(img_path, 'JPEG2000', compression_ratio=compression_ratio)
            temp_path = "temp_compressed.jp2"
            temp_files_to_clean = ["temp_input.png", temp_path]#deletes both i/p and o/p files

        #save and measure compress img
        compressed_img.save(temp_path)
        compressed_size = os.path.getsize(temp_path)#check new size
        print(f"Compressed Size: {compressed_size} bytes")
        
        ratio = compressed_size / original_size#find compression ratio
        print(f"Compression ratio= {ratio:.2f} of og size")

        #calculate quality metrics
        psnr, mse = self.calculate_psnr_mse(img_path, compressed_img)
        print(f"Quality Metrics - PSNR: {psnr:.2f} dB, MSE: {mse:.2f}")

        #delete temp files-changed it for jpeg2000
        for temp_file in temp_files_to_clean:
            if os.path.exists(temp_file):
                os.remove(temp_file)

        return compressed_img, ratio, psnr, mse#return img & ratio NEW: psnr and mse
    
    def calculate_psnr_mse(self, original_path, compressed_path):
        
        #self explanitory, calculates PSNR &mse
        #we need this for image quality assessment in compression
        #remember!!!:higher PSNR = better quality
        original_img = Image.open(original_path)#get OG img
        original = np.array(original_img)
        compressed = np.array(compressed_path)
       #checking that both images have same dimensions
        if original.shape != compressed.shape:
        #resize compressed to match original dimensions
            compressed = np.array(compressed_path.resize(original_img.size, Image.Resampling.LANCZOS))
# Convert to float for calculations
        original_float = original.astype(np.float64)
        compressed_float = compressed.astype(np.float64)
    
#calculate MSE
        mse = np.mean((original_float - compressed_float) ** 2)
    
#calculate PSNR
        if mse == 0:
            psnr = 100  # Perfect quality - identical images
        else:
            max_pixel = 255.0
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

        return psnr, mse
    
#Here we test how compression influences the AI's prediction performance   
    def test_prediction(self, img_path, standard='JPEG', quality=75, compression_ratio=20):#quality=75 is defult value
        print("Now testing prediction")

        from predictions import predict
        print(f"testing model w/ {standard} standard")
        #first we get the initial diagnosis for the imported,uncompressed img
        original_body_part = predict(img_path)#see predictions.py
        original_fracture = predict(img_path, model=original_body_part)
        print(f"initial predictions: {original_body_part} & {original_fracture}")

    # Compress with specified standard
        if standard == 'JPEG':
            compressed_img = self.compress_img(img_path, 'JPEG', quality)
            temp_path = f"temp_compressed_{quality}.jpg"
        else:  # JPEG2000
            compressed_img = self.compress_img(img_path, 'JPEG2000', compression_ratio=compression_ratio)
            temp_path = f"temp_compressed_{compression_ratio}.jp2"

        compressed_img.save(temp_path)
        compressed_body_part = predict(temp_path)
        compressed_fracture = predict(temp_path, model=compressed_body_part)
        print(f"Compressed ({standard}%): {compressed_body_part} - {compressed_fracture}")
        
        #here we compare answers to see if the compression confused the model.
        body_part_changed = (original_body_part != compressed_body_part)
        fracture_changed = (original_fracture != compressed_fracture)
    
        print(f"BodyPart answer change?: {'Yes..' if body_part_changed else 'No!'}")
        print(f"fracture detection changed?: {'yes..' if fracture_changed else 'No!'}")
    #!!!!should i add smth here that says what quality it broke at, if applicable?
        
        os.remove(temp_path)#cleanup images
    
        return body_part_changed or fracture_changed #indicate if any change occured
    
if __name__ == "__main__":#this only runs when compression.py is run:
    analyzer = CompressionAnalyzer()
    print("compression analyzer ready")
    #some images to use:
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
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"testing w/ {img_path}")
            print("-"*30)#just to make the ouput divided

            # Test JPEG at different quality levels
            print("\n JPEG COMPRESSION (DCT-based):")
            print("-"*30)
            for quality in [90, 75, 50, 25]:#different lvls to try
                print(f"\n quality level: {quality}%")
                compressed_img, ratio, psnr, mse = analyzer.test_single_image(img_path, standard='JPEG', quality=quality)
                print(f"file size reduced to {ratio*100:.1f}% of og")
                print(f"quality: PSNR {psnr:.2f} dB, MSE {mse:.2f}")
                #ok now ai model testing:
                #print("testing ai prediction!")
                #prediction_changed= analyzer.test_prediction(img_path, standard='JPEG', quality=quality)
                #print(f"model's prediction affected?:{'yes..' if prediction_changed else 'No!'}")
                
            # Test JPEG2000 at different quality levels
            print("\n JPEG2000 COMPRESSION (Wavelet-based):")
            print("-"*30)
            for compression_ratio in [10, 20, 40, 80]:#diff compression levels
                print(f"\nCompression ratio: {compression_ratio}:1")
                compressed_img, ratio, psnr, mse = analyzer.test_single_image(
                    img_path, standard='JPEG2000', compression_ratio=compression_ratio
                )
                print(f"File size: {ratio*100:.1f}% of original")
                print(f"Quality: PSNR {psnr:.2f} dB, MSE {mse:.2f}")
                #ok now ai model testing:
                # just trying one img for now
                #print("testing ai prediction!")
                #prediction_changed = analyzer.test_prediction(
                #    img_path, standard='JPEG2000', compression_ratio=compression_ratio
                #)
                #print(f"model's prediction affected?:{'yes..' if prediction_changed else 'No!'}")
            #break  # just trying one img for now
    else:
        print("no images found")