import io #to use bytes that are in memory i think
import os #to see file paths, sizes idk
from PIL import Image #this is basically img compression stuff from the pillow library we use
import numpy as np #for operations
import tensorflow as tf

#here we handle our experiments with testing topologies
#topologies will be comprised up of a standard + a quality lvl
#we decided to start with jpeg in last meeting as a first step
class CompressionAnalyzer:

    def __init__(self):#constructor"-autoruns when we 
                       #create a new obj in this class. gives initial obj state
    
        #list of standards, each obj needs a copy of this:
        self.supported_standards = ['JPEG']
        #print("initialized")

#this is a public intrface
    def compress_img(self, img_path,standard='JPEG',quality=90):
    #self: ref to object itself,imgpath:file path to img to compress
    #standard: string name of standard. quality: int value
        print(f"found: {img_path}")
        original_img = Image.open(img_path)#load img frm disk
        if standard =='JPEG':
         return self._jpeg_compress(original_img, quality)#calls JPEG fxn
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
    
#Here is where where testing is "streamlined"? idk if thsts a good word
    def test_single_image(self, img_path, quality=75):#quality=75 is defult value
        print(f"testing on {img_path}")
        original_size = os.path.getsize(img_path)#check OG file size
        print(f"Original size: {original_size} bytes")
        #compress it:
        compressed_img = self.compress_img(img_path, 'JPEG', quality)
        #saving temporarily
        temp_path = "temp_compressed.jpg"
        compressed_img.save(temp_path)
        compressed_size = os.path.getsize(temp_path)#check new size
        print(f"Compressed Size: {compressed_size} bytes")
        
        ratio = compressed_size / original_size#find compression ratio
        print(f"Compression ratio= {ratio:.2f} of og size")
        #delete temp files
        os.remove(temp_path)
        return compressed_img, ratio#return img & ratio

#Here we test how compression influences the AI's prediction performance   
    def test_prediction(self, img_path, quality=75):#quality=75 is defult value
        print("Now testing prediction")

        from predictions import predict
        print(f"testing model w/ {quality}% compression")
        #first we get the initial diagnosis for the imported,uncompressed img
        original_body_part = predict(img_path)#see predictions.py
        original_fracture = predict(img_path, model=original_body_part)
        print(f"initial predictions: {original_body_part} & {original_fracture}")

        compressed_img = self.compress_img(img_path, 'JPEG', quality)#now we compress
    
        #just saving the compressed one temporarily (clear it later)
        temp_path = f"temp_compressed_{quality}.jpg"
        compressed_img.save(temp_path)
        compressed_body_part = predict(temp_path)
        compressed_fracture = predict(temp_path, model=compressed_body_part)
        print(f"Compressed ({quality}%): {compressed_body_part} - {compressed_fracture}")
        
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
            for quality in [90, 75, 50, 25]:#different lvls to try
                print(f"\n quality level: {quality}%")
                compressed_img, ratio = analyzer.test_single_image(img_path, quality)
                print(f"file size reduced to {ratio*100:.1f}% of og")
                #ok now ai testing:
                print("testing ai prediction!")
                prediction_changed= analyzer.test_prediction(img_path,quality)
                print(f"model's prediction affected?:{'yes..' if prediction_changed else 'No!'}")
            break  # just trying one img for now
    else:
        print("no images found")