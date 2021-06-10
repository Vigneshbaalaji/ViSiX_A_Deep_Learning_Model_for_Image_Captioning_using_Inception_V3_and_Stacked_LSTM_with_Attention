import cv2
import pyrebase
import numpy as np
from PIL import Image
import streamlit as st
from io import BytesIO
from google.cloud import firestore
from firebase_admin import db
import os
from firebase import Firebase
from google.cloud import firestore
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

config = {
    "apiKey": "AIzaSyBuUAKW5RW4N6rctX8VBORZSAB43RiLv40",
    "authDomain": "osvkmvisix.firebaseapp.com",
    "databaseURL": "https://osvkmvisix-default-rtdb.firebaseio.com",
    "projectId": "osvkmvisix",
    "storageBucket": "osvkmvisix.appspot.com",
    "messagingSenderId": "866144431462",
    "appId": "1:866144431462:web:c141344ca58de27626aca7",
    "measurementId": "G-F1CW4D9S7Z"

}

STYLE = """
<style>
img {
    max-width:100%
}
</style>

"""
st.title("VISIX Application for Image captioning")
filename = "temp.jpg"


uploaded_file = st.file_uploader("Welcome :) please upload an image ", type=["jpg","png"])
show_file = st.empty()


if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    x = uploaded_file.name
    print(x)
    filename = x[:5] + filename
    cv2.imwrite(filename, image)
    my_img = filename
    firebase = pyrebase.initialize_app(config)
    storage = firebase.storage()
    storage.child(my_img).put(my_img)
    show_file.image(image, channels="BGR")

 # ref = db.collection('captions').doc('Image_caption').get('Captions')
 #  print(ref)    

else:
    show_file.info("Please upload the file in the Allowed formats : {}".format(' '.join(["JPG"," PNG"])))



# Use a service account
#cred = credentials.Certificate('path/to/serviceAccount.json')
cred = credentials.Certificate('osvkmvisix-firebase-adminsdk-2dhok-99db1f007e.json')
#firebase_admin.initialize_app(cred, {
#    'databaseURL': 'https://osvkmvisix-default-rtdb.firebaseio.com'})

ref = db.reference('restricted_access/secret_document')
print(ref.get())

db = firestore.client()

if st.button('Generate captions!'):
    docs = db.collection(u'captions').stream()

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')
        y = doc.to_dict()['Images']
        print(y)
        w = my_img.lower()
        if (y == w):
            z = doc.to_dict()['Captions']
            print(z)
            st.text("Generated caption : ")
            st.text(z)
            st.balloons() 
    st.success("Click again to retry or try a different image by uploading")
   
    
