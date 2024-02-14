from flask import Flask, render_template, request,send_file,url_for
import nibabel as nib
import os
from werkzeug.utils import secure_filename
from transform import test_transforms
from resampling import resample_nifti
from visualizeSegResNet import SegResNet_output, model_SegResNet
from visualizeUnet import UNet_output, model_Unet
app=Flask(__name__)
UPLOAD_FOLDER='./uploads'
ALLOWED_EXTENSION={'nii.gz'}

app.config['UPLOAD_FOLDER']=UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = 'static'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', error='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(file_path)
        
        file.save(file_path)
        nifti_img = nib.load(file_path)
        data = nifti_img.get_fdata()

    # Get the current number of slices
        num_of_slice = data.shape[-1]
        if(num_of_slice>50):
           resample_nifti(file_path, 50)

        # Visualize the segmentation output for all slices
        img_SegResNet = SegResNet_output(volume=file_path, model=model_SegResNet)
        img_Unet=UNet_output(volume=file_path, model=model_Unet)


        # Return the list of segmentation output images as a response
        return render_template('index.html', segmentation_output_list1=img_SegResNet,segmentation_output_list2= img_Unet)
    
         

    return render_template('index.html', error='Invalid file format')




if __name__ == '__main__':
    app.run(debug=True)

    