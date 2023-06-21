# -*- coding: utf-8 -*-
"""
Created on Sat Jun 17 12:00:52 2023

@author: hp
"""
import os
from flask import Flask,request,render_template,send_file,session

from werkzeug.utils import secure_filename
import io
import base64
import cv2
import numpy as np
import json
# ...


app = Flask(__name__)

app.secret_key = 'abcd'

@app.route('/')
def index():
    session.clear()  # Clear the session data
    #num_contours=session.get('num_contours')
    return render_template('contour_visualization.html')

@app.route('/preprocess_image', methods=['POST'])
def preprocess_image():
    if request.method == 'POST':
        # Retrieve the uploaded image file
        image_file = request.files['image']
        filename = secure_filename(image_file.filename)
        image1 = cv2.imread('images/' + filename)
        # Convert the image to grayscale
        gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        # Apply a threshold to create a binary image
        ret, binary_mask = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Find contours in the binary image
        contours, hierarchy = cv2.findContours(binary_mask,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
         
        
        destination_dir = 'upload'
        os.makedirs(destination_dir, exist_ok=True)  # Create directory if it doesn't exist
        image_path = os.path.join(destination_dir, filename + '.png')
        cv2.imwrite(image_path, binary_mask) 
        
        destination_dir_org = 'static/org_image'
        os.makedirs(destination_dir_org, exist_ok=True)  # Create directory if it doesn't exist
        image_path_org = os.path.join(destination_dir_org, filename)
        cv2.imwrite(image_path_org, image1) 
        
        # Convert the contours to nested lists
        contours_list = [contour.tolist() for contour in contours]
        # Store the necessary data in the session
        session['image1'] = image_path
        session['image_org']=image_path_org
        session['image_name']=filename
        session['num_contours'] = len(contours_list)

        #num_contours=len(contours_list)
        return render_template('contour_visualization.html', num_contours=session['num_contours'])

    return render_template('contour_visualization.html')


@app.route('/mask_image', methods=['POST'])
def mask_image():
    if request.method == 'POST':
        # Retrieve the user-submitted number of contours
        num_contours = session.get('num_contours')
        factors = []
        for i in range(num_contours):
            factor = request.form.get(f"factor_{i}")
            print(f"factor_{i}:", factor) 
            if factor is None:
                return f"factor_{i} not provided"
            factors.append(float(factor))
        if len(factors) != num_contours:
            return "Number of factors doesn't match the number of contours"

        factors = [float(factor) for factor in factors]

        # Retrieve the necessary data from the session
        image1 = session.get('image1')
        image2 = cv2.imread(image1)
        img = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #Reverting the original image back to BGR so we can draw in colors
        img_c = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
        img_c1=cv2.drawContours(img_c, contours, -1, (0, 255, 0), 3)
        if image2 is None or contours is None:
         
            return "Image or contours data not found in the session"
        
        filename = session.get('image_name')
        # Perform the remaining operations
        # Iterate over each contour
        
        L=[]
        W=[]
        for i, contour in enumerate(contours):
            # Create a blank image for the contour
            contour_image = np.zeros_like(image2)
           
            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
            x, y, width, height = cv2.boundingRect(contour)
            #perimeter = cv2.arcLength(contour, True)

            # Display the measured dimensions
            w=width*factors[i]
            print("Width in cm: {} cm".format(w))
            #print("Height: {} pixels".format(height))
            h=height*factors[i]


            L.append(h)
            W.append(w)
            # Draw the contour and bounding box on the image (optional)
            #cv2.drawContours(contour_image, [contour], 0, (0, 255, 0), 2)
            cv2.rectangle(contour_image, (x, y), (x + width, y + height), (0, 0, 255), 2)

            # Draw the current contour on the image
            cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)
            # Create a text string with the dimensions
            
          
            # Save the modified contour image with a unique filename
            fname ='static/image'
            os.makedirs(fname, exist_ok=True)  # Create directory if it doesn't exist
            image_path = os.path.join(fname, f'contour_{i}' + '.png')
            cv2.imwrite(image_path, contour_image)
            
        return render_template('contour_visualization.html', num_contours=num_contours,length=L,width=W,filename=filename)

if __name__ == '__main__':
    app.run(debug=True)