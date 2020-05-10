from flask import Flask
from flask import render_template
from flask import request
from flask import flash
import models as dbHandler
from config import ACCESS_KEY,SECRET_KEY
import boto3
from werkzeug.utils import secure_filename
import test
# Toast messages
from flask_toastr import Toastr
toastr = Toastr()

app = Flask(__name__)

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

# Upload page for user to upload images on S3
@app.route('/uploader', methods=['POST', 'GET'])
def uploader():
    if request.method == "POST":
        try:
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

# Run median filter
@app.route('/median', methods=['POST', 'GET'])
def median():

    # # Compute RMSE
    # print("RMSE: ", rmse(denoised, result))

    # # Compute UQI
    # print("UQI: ", uqi(denoised,result))

    # # Compute PSNR
    # print("PSNR: ", psnr(denoised, result))

    # # Compute SSIM
    # print("SSIM:", ssim(denoised, result))
    # return test.t
    flash('Toastr works')
    # return '<h2>Median</h2>'

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')