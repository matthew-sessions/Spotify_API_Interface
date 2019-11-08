from flask import Flask, render_template, request, url_for, redirect, jsonify
from api_call import *

from flask_cors import CORS
import pygal


app = Flask(__name__)
app.static_folder = 'static'
CORS(app)

@app.route('/')
def home():
    return(render_template('home.html'))

@app.route('/search')
def search():
    name = request.args.get('search')
    songs = song_search(name)
    return(render_template('search.html', title='Current user', search=songs, name=name))

@app.route('/recs/<value>')
def recs(value):
    songs = song_recs(value)
    graph = songs[0]['graph_uri']
    feats = songs[1]
    li = [i['id'] for i in songs[2]]
    name = 'the song you selected.'
    return(render_template('recs.html', title='Current user',
                                          search=songs[2],
                                          graph=graph,
                                          feats=feats,
                                          li=li,
                                          name = name))

@app.route('/recs_mood')
def recs_mood():
    acousticness = request.args.get('acousticness')
    danceability = request.args.get('danceability')
    duration_ms = request.args.get('duration_ms')
    energy = request.args.get('energy')
    instrumentalness = request.args.get('instrumentalness')
    key = request.args.get('key')
    liveness = request.args.get('liveness')
    loudness = request.args.get('loudness')
    mode = request.args.get('mode')
    speechiness = request.args.get('speechiness')
    tempo = request.args.get('tempo')
    time_signature = request.args.get('time_signature')
    valence = request.args.get('valence')
    playlist = request.args.get('playlist')

    name = 'the mood values you selected.'

    songs = song_recs_mood(acousticness,
                             danceability,
                             duration_ms,
                             energy,
                             instrumentalness,
                             key,
                             liveness,
                             loudness,
                             mode,
                             speechiness,
                             tempo,
                             time_signature,
                             valence,
                             playlist)
    graph = songs[0]['graph_uri']
    feats = songs[1]
    li = [i['id'] for i in songs[2]]
    return(render_template('recs.html', title='Current user',
                                          search=songs[2],
                                          graph=graph,
                                          feats=feats,
                                          li=li,
                                          name = name))


if __name__ == '__main__':
    app.run()
