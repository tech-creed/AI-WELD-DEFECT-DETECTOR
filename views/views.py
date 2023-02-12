from flask import make_response,Flask, flash, redirect, render_template, request, url_for, session
from app import *
from utilss.models import *

import os
from os import walk

import bcrypt

from utilss.upload import allowed_file, erase_dir, file_saver

#-------------------------------------------------------------------------#
# Main Page
@app.route('/')
def home():
    return render_template('index.html')

#SignUp
@app.route('/register', methods=["POST", "GET"])
def signup():
    message = ''
    if "email" in session:
        return redirect('/dashboard')

    if request.method == "POST":
        user = request.form.get("name")
        email = request.form.get("email")
        
        password1 = request.form.get("password")
        password2 = request.form.get("cpassword")

        user_found = db.user.find_one({"name": user})
        email_found = db.user.find_one({"email": email})

        if user_found:
            message = 'There already is a user by that name'
            return render_template('login.html', message=message)
        if email_found:
            message = 'This email already exists in database'
            return render_template('login.html', message=message)
        if password1 != password2:
            message = 'Passwords should match!'
            return render_template('login.html', message=message)

        else:
            hashed = bcrypt.hashpw(password2.encode('utf-8'), bcrypt.gensalt())
            user_input = {'name': user, 'email': email, 'password': hashed}
            db.user.insert_one(user_input)
            
            user_data = db.user.find_one({"email": email})
            new_email = user_data['email']
   
            return redirect('/dashboard')

    return render_template('register.html')

#Login
@app.route('/login', methods=["POST", "GET"])
def login():
    message = 'Please login to your account'
    if "email" in session:
        return redirect('/dashboard')

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
       
        email_found = db.user.find_one({"email": email})
        if email_found:
            email_val = email_found['email']
            passwordcheck = email_found['password']
            
            if bcrypt.checkpw(password.encode('utf-8'), passwordcheck):
                session["email"] = email_val
                return redirect('/dashboard')
            else:
                if "email" in session:
                    return redirect(url_for("logged_in"))
                message = 'Wrong password'
                return render_template('login.html', message=message)
        else:
            message = 'Email not found'
            return render_template('login.html', message=message)

    return render_template('login.html', message=message)


@app.route("/logout", methods=["POST", "GET"])
def logout():
    if "email" in session:
        session.pop("email", None)
        return redirect('/login')
    else:
        return redirect('/login')


#-------------------------------------------------------------------------#
# Dashboard
@app.route("/dashboard", methods=["POST", "GET"])
def upload():
    if "email" in session:
        return render_template('upload.html')
    else:
        return redirect('/login')

#-------------------------------------------------------------------------#
#Upload Get
@app.route('/upload', methods=['GET'])
def UploadGet():
    if "email" in session:
        return render_template('upload.html')
    else:
        return redirect('/login')

#Upload Post
@app.route('/upload', methods=['POST'])
def UploadPost():
    print('!!')
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '' :
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        erase_dir()
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename)) 
        imgName = file_saver(file.filename, request.form["Id"])

        make_augumentation(imgName,request.form["Id"])
        detectDefect(imgName,request.form["Id"])
        
    else:
        flash('Allowed image types are -> png, jpg, jpeg, bmp')
        return redirect(request.url)
        
    return redirect(f'/result/{request.form["Id"]}')

#Upload Get
@app.route('/result/<id>', methods=['GET'])
def Result(id):
    if "email" in session:
        return render_template('result.html', id=id)
    else:
        return redirect('/login')