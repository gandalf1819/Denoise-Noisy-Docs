from flask import Flask, render_template, request, flash
from config import ACCESS_KEY,SECRET_KEY
import boto3
# from werkzeug import secure_filename
from werkzeug.utils import secure_filename

app = Flask(__name__)

app.secret_key = 'some_random_key'
bucket_name = "denoise-docs"

s3 = boto3.client(
   "s3",
   aws_access_key_id=ACCESS_KEY,
   aws_secret_access_key=SECRET_KEY
)
bucket_resource = s3

@app.route("/", methods=['post', 'get'])
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
                # return("<h1>upload successful<h1>")
                flash('Upload Successful!')
        except Exception as e:
            return (str(e))
    return render_template("upload.html")


if __name__ == '__main__':
    app.run(debug=True)