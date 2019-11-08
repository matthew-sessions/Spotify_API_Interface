import requests



def song_search(url):
    get = requests.get('https://spotify-api-helper.herokuapp.com/songs/DReaI4d55IIaiD6P9/' + url)
    songs = get.json()
    return(songs)

def song_recs(url):
    get = requests.get(f'https://spotify-api-helper.herokuapp.com/playlist_recs/DReaI4d55IIaiD6P9?playlist=[{url}]')
    songs = get.json()
    return(songs)


def song_recs_mood(acousticness,danceability,duration_ms,energy,instrumentalness,key,liveness,loudness,mode,speechiness,tempo,time_signature,valence, playlist):
    get = requests.get(f'https://spotify-api-helper.herokuapp.com/playlist_mood_recs/DReaI4d55IIaiD6P9?acousticness={acousticness}&danceability={danceability}&duration_ms={duration_ms}&energy={energy}&instrumentalness={instrumentalness}&key={key}&liveness={liveness}&loudness={loudness}&mode={mode}&speechiness={speechiness}&tempo={tempo}&time_signature={time_signature}&valence={valence}&playlist={playlist}')
    songs = get.json()
    return(songs)
