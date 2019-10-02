import base64
import numpy as np
import io
#from PIL import Image
import keras
from keras import backend as k
from keras.models import Sequential
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator 
from keras.preprocessing.image import img_to_array
from flask import request
from flask import jsonify
from flask import Flask 
from flask import render_template 
import tensorflow as tf
from scipy.io import wavfile
from keras.models import model_from_json
from scipy.io import wavfile
#import IPython.display as ipd
from io import StringIO
import tempfile
from flask import make_response
import random
from keras import backend
import os


app = Flask("Prueba")


def get_model():

	global model
	model = load_model('i_working_weights.h5')
	'''linea extra'''
	model._make_predict_function()

	print(" * Modelo cargado")

def preprocess_image(image,target_size):

	if image.mode != "RGB":
		image =image.convert("RGB")
	image = image.resize(target_size)
	image = img_to_array(image)
	image = np.expand_dims(image,axis=0)

	return image

# Main functions needed

# Normalize data to range (-1, 1)
def norm(data):
    max_data = np.max(data)
    min_data = np.min(data)
    if abs(min_data) > max_data:
        max_data = abs(min_data)
    data = data / max_data
    return data

# Round every sample to 3 decimals
def rounding(data):
    for x in range(0, data.size):
        data[x] = round(data[x], 3)
    return data

# Damages to audio to the case of loss of information
def br_incompletion(data, p_in):
    
    # The positions where I want to place the 0's
    maska = np.random.choice([0, 1], size=data.size, p=[1-p_in, p_in])
    
    data = np.ma.array(data, mask=maska, fill_value=0)
    resultado = data.filled()

    return resultado

# Damages the audio by adding GWN
def br_random(data, p_no):
    
    mu = 0      # mean
    sigma = 0.15 # standard deviation
    
    mask = np.random.choice([0, 1], size=data.size, p=[1-p_no, p_no])
    mask = np.array(mask, dtype=bool)
    
    sine = np.random.normal(mu, sigma, data.size)
    
    broken = data.copy()
    broken[mask] = sine[mask]
    
    return broken

# Damages the audio by decreasing the quality to 8000 Hz
def br_low_q(data):
    
    # The low q audio from the original (half length)
    half_d = data[1::2]

    return half_d

# Addes 0s to the low q audio so it has the same length
def fx_low_q(data):
    
    # The low q audio but with 0's. (same length)
    temp = np.zeros(data.size)
    full_d = np.empty(data.size*2)
    
    full_d[0::2] = data
    full_d[1::2] = temp
    
    return full_d
# Receives the original audio
# Function that somehow creates the numpy array from whatever it gets
def audio2numpy(audio):
    
    # In case it receives a wav file from a path
    recording = norm(audio)
    recording = rounding(recording)
    
    return recording

def damage_option(option, recording):
    
    p_in = 0.7
    p_no = 0.3
    
    global model
        
    if option == 1:
        print('\nIncompletion selected')
        broken_file = br_incompletion(recording, p_in)
        # Call function to load the model
        path_m = 'static/models/i_working_model.json'
        path_w = 'static/models/i_working_weights.h5'
        
    elif option == 2:
        print('\nRandom Noise selected')
        broken_file = br_random(recording, p_no)
        # Call function to load the model
        path_m = 'static/models/r_working_model.json'
        path_w = 'static/models/r_working_weights.h5'
        
    elif option == 3:
        print('\nLow Quality selected\n')
        # Call function to load the model
        path_m = 'static/models/lq_working_model.json'
        path_w = 'static/models/lq_working_weights.h5'
        broken_file = br_low_q(recording)
        broken_file = fx_low_q(broken_file)
        
    elif option == 4:
        print('\nOriginal audio selected')
        broken_file = recording
        # Call function to load the model
        path_m = 'static/models/r_working_model.json'
        path_w = 'static/models/r_working_weights.h5'
        
    broken_file = rounding(broken_file)

    json_file = open(path_m, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(path_w)
    model.compile(optimizer='adam', loss='mse', metrics = ['accuracy'])
    
    return broken_file

def cut_audio(broken_file):
    
    br_temp = list()
    leng=32000
    # Eye eye...
    # Lets calculate the next following array that's divisible in our files len
    orig_len = len(broken_file)
    # How many times (or subarrays) +1 for the next one (will contain zeros)
    divs = ((orig_len//32000)+1)
    # Here we create a np zeros of the calculated len is divisible
    temp = np.zeros(divs*32000)
    # Now we substract the orig len so we have the remaining part that completes the original
    temp = temp[:len(temp)-orig_len]
    all_broken = np.concatenate((broken_file, temp), axis=0)
    
    for i in range(divs):
        print(all_broken[i*leng:(i+1)*leng])
        br_temp.append(all_broken[i*leng:(i+1)*leng])
    
    return br_temp,orig_len,divs

def waverinet_restore(br_temp, orig_len,divs):
    
    out_n = list()
    restored = list()

    for i in range(divs):
        in_temp = np.reshape(br_temp[i], (1, np.shape(br_temp[i])[0], 1))
        out_temp = model.predict(in_temp)
        out_temp = np.reshape(out_temp, (np.shape(out_temp)[1]))
        restored = np.concatenate((restored, out_temp), axis=0)

    restored = restored[:orig_len]
    
    return restored

'''print("Cargando modelo")
get_model()'''

@app.route('/')
def index():   
    return render_template("index.html")
    

if __name__ == '__main__':
	app.run(host='127.0.0.1', port=5000, debug=True)

@app.route("/waverinetdamage",methods=['POST'])


def predict():

    message = request.get_json(force=True)

    encoded = message['audio']

    option = int(message['tipo'])

    boolrandom = int(message['randomsi_no'])

    print(type(encoded))
    print('Es un fichero de local o random 1 random 0 local: ', boolrandom)   

    global nombred
    if (boolrandom==1):

        recording = wavfile.read("static/"+encoded)

        nombred = random.randint(1,10000)

        nombred = str(nombred)

        nombred = nombred+".wav"

    else:

        decoded = base64.b64decode(encoded)

        recording = wavfile.read(io.BytesIO(decoded))

        nombred = random.randint(1,10000)

        nombred = str(nombred)

        nombred = nombred+".wav"

    original_audio = audio2numpy(recording[1])
    broken_file = damage_option(option, original_audio) 

    broken_file = np.array(broken_file, dtype= np.float32)
    
    wavfile.write("static/"+nombred, 16000, broken_file)

 
    br_temp,orig_len,divs = cut_audio(broken_file)

    global restored_audio
    restored_audio = waverinet_restore(br_temp, orig_len,divs)

    backend.clear_session()

    response = {
    	'prediction':{
            'nombre_damaged':nombred
     	}
    }


    return jsonify(response)


@app.route("/waverinetrestore",methods=['POST'])


def predict2():

    restored_file = np.array(restored_audio, dtype= np.float32)
    global nombrer

    nombrer = random.randint(1,10000)

    nombrer = str(nombrer)

    nombrer = nombrer+"restored"+".wav"
   
    wavfile.write("static/"+nombrer, 16000, restored_file)

    backend.clear_session()
    
    response = {
        'prediction':{
            'nombre_restored':nombrer
        }
    }

    return jsonify(response)

@app.route("/waverinetreset",methods=['POST'])

def predict3():

    os.remove("static/"+nombrer)
    os.remove("static/"+nombred)

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r
	

	



