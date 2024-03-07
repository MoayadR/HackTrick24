import numpy as np
from numpy import load
import requests
from keras.models import load_model
import pickle
from LSBSteg import decode

api_base_url = "http://3.70.97.142:5000/eagle"
team_id = "R5VjcNw"
Tx = 1998
Nfreq = 101


def load_scaler():
    '''
    Load your scaler here and return it
    '''
    with open('model/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def load_tf_model():
    '''
    Load your model here and return it
    '''
    model = load_model('model/model_footprints.h5')
    # model = load_model('model/model.h5')
    return model

scaler = load_scaler()
model = load_tf_model()

def preprocess_footprint(footprint):
    '''
    Preprocess the footprint to be in the right format for your model
    '''
    data = np.array(footprint)
    
    spectrograms = np.nan_to_num(data)
    spectrograms_scaled = scaler.transform(spectrograms.reshape(-1, Tx * Nfreq)).reshape(-1, Tx, Nfreq, 1)
    return spectrograms_scaled

def init_eagle(team_id):
    '''
    In this fucntion you need to hit to the endpoint to start the game as an eagle with your team id.
    If a sucessful response is returned, you will recive back the first footprints.
    '''
    body = {"teamId": team_id}
    footprints = requests.post(api_base_url + "/start", json = body)
    return footprints

def select_channel(footprint):
    '''
    According to the footprint you recieved (one footprint per channel)
    you need to decide if you want to listen to any of the 3 channels or just skip this message.
    Your goal is to try to catch all the real messages and skip the fake and the empty ones.
    Refer to the documentation of the Footprints to know more what the footprints represent to guide you in your approach.        
    '''
    footprint = np.array(footprint)
    footprint = preprocess_footprint(footprint)
    y = model.predict(footprint)
    real = [True if i > 0.5 else False for i in y]
    return real[0]


def skip_msg(team_id): # can lead to pure string response
    '''
    If you decide to NOT listen to ANY of the 3 channels then you need to hit the end point skipping the message.
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    body = {"teamId": team_id}
    next_footprint = requests.post(api_base_url + "/skip-message", json = body)
    # next_footprint = next_footprint["footprint"]  # this will be a dict {”1”: spectrogram1, ”2”:spectrogram2, ”3”:spectrogram3 }
    return next_footprint
    

def request_msg(team_id, channel_id):
    '''
    If you decide to listen to any of the 3 channels then you need to hit the end point of selecting a channel to hear on (1,2 or 3)
    '''
    body = {"teamId": team_id, "channelId": channel_id}
    encoded_msg = requests.post(api_base_url + "/request-message", json = body).json()
    encoded_msg = np.array(encoded_msg["encodedMsg"]) # this will be a np.array  [[0.2 0.4 0.6] [0.3 0.5 0.7], [0.1 0.8 0.9]]
    return encoded_msg

def submit_msg(team_id, decoded_msg): # can lead to pure string response
    '''
    In this function you are expected to:
        1. Decode the message you requested previously
        2. call the api end point to send your decoded message  
    If sucessful request to the end point , you will expect to have back new footprints IF ANY.
    '''
    body = {"teamId": team_id, "decodedMsg": decoded_msg}
    next_footprint = requests.post(api_base_url + "/submit-message", json = body)
    # next_footprint = next_footprint["footprint"]  # this will be a dict {”1”: spectrogram1, ”2”:spectrogram2, ”3”:spectrogram3 }
    return next_footprint

def end_eagle(team_id):
    '''
    Use this function to call the api end point of ending the eagle  game.
    Note that:
    1. Not calling this fucntion will cost you in the scoring function
    '''
    body = {"teamId": team_id}
    response = requests.post(api_base_url + "/end-game", json = body)
    return response

def rec(footprints,first=False):
    if footprints.text == 'End of message reached':
        return
    elif first == True:
        footprints = footprints.json()['footprint']
    else:
        footprints = footprints.json()['nextFootprint']
        
    for channel_id, footprint in footprints.items():
        if select_channel(footprint):
            encoded_msg = request_msg(team_id, int(channel_id))
            decoded_msg = decode(encoded_msg)
            print('decoded_msg: ', decoded_msg)
            res = submit_msg(team_id, decoded_msg)
            # if res == 'End of message reached' break else request next message 
            return rec(res)
        
    res = skip_msg(team_id) 
    return rec(res)

def sol(input_footprints):
    global footprints
    footprints = input_footprints.json()['footprint']
    first = True
    while True:
        skip = True
        if not first and (footprints.text == 'End of message reached'):
            break
        elif not first:
            footprints = footprints.json()['nextFootprint']
        first = False
        for channel_id, footprint in footprints.items():
            if select_channel(footprint):
                skip = False
                encoded_msg = request_msg(team_id, int(channel_id))
                decoded_msg = decode(encoded_msg)
                print('decoded_msg: ', decoded_msg)
                res = submit_msg(team_id, decoded_msg)
                # if res == 'End of message reached' break else request next message 
                footprints = res
                break
        if skip:
            footprints = skip_msg(team_id)

def get_remaining_attempts(team_id):
    req = requests.post("http://13.53.169.72:5000/attempts/student", json = {"teamId": team_id})
    print(req.json())


def submit_eagle_attempt(team_id):
    '''
     Call this function to start playing as an eagle. 
     You should submit with your own team id that was sent to you in the email.
     Remeber you have up to 15 Submissions as an Eagle In phase1.
     In this function you should:
        1. Initialize the game as fox 
        2. Solve the footprints to know which channel to listen on if any.
        3. Select a channel to hear on OR send skip request.
        4. Submit your answer in case you listened on any channel
        5. End the Game
    '''
    footprints = init_eagle(team_id)
    rec(footprints,first=True)
    # sol(footprints)
    res = end_eagle(team_id)
    print('res: ', res.text)


# submit_eagle_attempt(team_id)

get_remaining_attempts(team_id)