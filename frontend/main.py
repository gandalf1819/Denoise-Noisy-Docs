from flask import Flask
from flask import render_template
from flask import request
from flask import flash
import models as dbHandler
from config import ACCESS_KEY,SECRET_KEY
import boto3
import cv2
import os
from werkzeug.utils import secure_filename

# Import method files
# import median_filter as mf

# Add folder path
IMAGE_FOLDER = os.path.join('static', 'results/median-results')


# Toast messages
from flask_toastr import Toastr
toastr = Toastr()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = IMAGE_FOLDER

# Initialize toastr
toastr.init_app(app)

# S3 bucket configuration for AWS
app.secret_key = 'some_random_key'
bucket_name = "denoise-docs"

s3 = boto3.client(
   "s3",
   aws_access_key_id=ACCESS_KEY,
   aws_secret_access_key=SECRET_KEY
)
bucket_resource = s3

# Login page for Noisy Docs Denoiser
@app.route('/', methods=['POST', 'GET'])
def home():
    if request.method=='POST':
        username = request.form['username']
        password = request.form['password']
        dbHandler.insertUser(username, password)
        users = dbHandler.retrieveUsers()
        return render_template('index.html', users=users)
    else:
        return render_template('index.html')

# Test routing
@app.route('/burger')
def burger():
    return '<h2>Burger King!</h2>'

#background process happening without any refreshing
@app.route('/background_process_test')
def background_process_test():
    print("Hello")
    return 'nothing'

# Upload page for user to upload images on S3
@app.route('/uploader', methods=['POST', 'GET'])
def uploader():
    if request.method == "POST":
        try:
            # Upload image on S3 bucket
            img = request.files['img']
            filename = ''
            if img:
                filename = secure_filename(img.filename)
                img.save(filename)
                bucket_resource.upload_file(
                    Bucket = bucket_name,
                    Filename=filename,
                    Key=filename
                )
                # return "<h1>upload successful<h1>"
                flash('Upload successful!')
        except Exception as e:
            return (str(e))
    return render_template('upload.html')

# Run Adaptive thresholding
@app.route('/adaptive', methods=['POST', 'GET'])
def adaptive():

    return render_template('adaptive.html')

# Run Median filtering
@app.route('/median', methods=['POST', 'GET'])
def median():
    # Upload image on S3 bucket
    # if request.method == "POST":
    #     try:
    #         # Upload image on S3 bucket
    #         img = request.files['img']
    #         print(type(img))
    #         filename = ''
    #         if img:
    #             filename = secure_filename(img.filename)
    #             # Store the file locally 
    #             img.save(filename)

    #             # Get the file name
    #             bucket_resource.upload_file(
    #                 Bucket = bucket_name,
    #                 Filename=filename,
    #                 Key=filename
    #             )
    #             # return "<h1>upload successful<h1>"
    #             flash('Upload successful!')

    #     except Exception as e:
    #         return (str(e))

    # ## Median Filtering logic from median.py
    # # Read image uploaded by user recently using 'filename' variable
    # user_img = cv2.imread(filename,  cv2.IMREAD_GRAYSCALE)

    # # Apply median filtering from median.py script
    # result, background = mf.median_subtract(user_img)

    # # Compute rmse, uqi, psnr, ssim for the uploaded image from median.py
    # rmse = mf.m_rmse
    # uqi = mf.m_uqi
    # psnr = mf.m_psnr
    # ssim = mf.m_ssim
    
    # # Dump the result after median filtering 
    # median_output = 'median_' + filename
    # cv2.imwrite(median_output, result)

    # # Store the output image on S3 - median_output
    # try:
    #     bucket_resource.upload_file(
    #         Bucket = bucket_name,
    #         Filename=median_output,
    #         Key=median_output
    #     )

    #     flash('Result saved successfully on AWS!')
    # except Exception as e:
    #     return (str(e))

    # # Display image in median.html
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'clean.png')

    return render_template('median.html', user_image = image_path)

# Run Edge filtering
@app.route('/edge', methods=['POST', 'GET'])
def edge():
    return render_template('edge.html')

# Run Autoencoder
@app.route('/autoencoder', methods=['POST', 'GET'])
def autoencoder():
    return render_template('autoencoder.html')

# Run Regression
@app.route('/regression', methods=['POST', 'GET'])
def regression():
    return render_template('regression.html')

# Run median filter
@app.route('/test', methods=['POST', 'GET'])
def test():

    flash('Toastr works')
    return render_template('test.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')