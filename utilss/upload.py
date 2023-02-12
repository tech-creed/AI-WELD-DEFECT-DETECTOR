from os import walk
from PIL import Image
from app import *
import os

# Helper Constant and Function to Delete the Old Xray
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

def file_saver(filename, patient_id):
    os.makedirs((os.path.join(app.config['UPLOAD_FOLDER'],patient_id+'/')),exist_ok=True)
    file = os.path.splitext(filename)
    if file[1] == ".bmp":
        Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).save(os.path.join(app.config['UPLOAD_FOLDER'],patient_id+'/'+filename.split('.')[0]) + '.png')
        Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).save(os.path.join(app.config['UPLOAD_FOLDER'],patient_id+'/') + 'org.png')
        os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).save(os.path.join(app.config['UPLOAD_FOLDER'],patient_id+'/'+filename.split('.')[0]) + '.png')
        Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename)).save(os.path.join(app.config['UPLOAD_FOLDER'],patient_id+'/') + 'org.png')
        if file[1] != '.png':
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return str(os.path.join(app.config['UPLOAD_FOLDER'],patient_id+'/'+filename.split('.')[0]) + '.png')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def erase_dir():
    f = next(walk('static/uploads/'), (None, None, []))[2]
    for ele in f:
        os.remove('static/uploads/'+ele)