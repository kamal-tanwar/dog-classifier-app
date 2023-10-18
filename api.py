import os
import secrets
from PIL import Image
from flask import Flask, render_template, request, flash

from predict import predict_breed_transfer


app = Flask(__name__)
UPLOAD_FOLDER = "./static/uploads"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'} 
MODEL = None
device = "cuda"

app.config['SECRET_KEY'] = 'your_secret_key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_picture(image_file):
    random_hex = secrets.token_hex(8)
    _, f_ext = os.path.splitext(image_file.filename)
    picture_fn = random_hex + f_ext
    picture_path = os.path.join(UPLOAD_FOLDER, picture_fn)

    output_size = (256, 256)
    i = Image.open(image_file)
    i.thumbnail(output_size)
    i.save(picture_path)

    return picture_path


@app.route("/", methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file and allowed_file(image_file.filename):
            image_location = save_picture(image_file)
            pred = predict_breed_transfer(image_location)
            return render_template('index.html', prediction=pred, image_loc=image_location)
        else:
            flash('Invalid file type. Please upload an image with a valid extension (png, jpg, jpeg).', 'danger')
    return render_template('index.html', prediction='', image_loc=None)


if __name__ == "__main__":      
    app.run(port=12000, debug=True)