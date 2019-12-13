## View the API interface app here: https://spotirecer.herokuapp.com/


# Spotify Song Recommender

The scope of this project was to build an application that uses the Spotify API along with Machine Learning techniques to recommend songs that users are likely to enjoy. 

### Available Data

We initially took a look at the [Spotify Audio Feature dataset on Kaggle.](https://www.kaggle.com/tomigelo/spotify-audio-features) This dataset has 17 features in total, 13 of which are actual audio features. 

![alt text](https://github.com/bw-spotify-oct/ds/blob/master/img/feat.png "Logo Title Text 1")

For the purpose of creating a model, we dropped “artist_name”, “track_name”, and “popularity”. Popularity is an interesting feature due to the fact that it does hold predictive power but is not entirely useful for our model because it is not a constant variable. Songs that have a high popularity score today may not have the same score tomorrow.

### Here is a rundown of the features we used:

**duration_ms** - The duration of the track

**key** - The estimated overall key of the track.

**mode** - Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived.

**time_signature** - An estimated overall time signature of a track

**acousticness** - A confidence measure from 0.0 to 1.0 of whether the track is acoustic

**danceability** - Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity.

**energy** - Energy is a measurement that represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy.

**instrementalness** - Predicts whether a track contains no vocals. “Ooh” and “aah” sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly “vocal”.

 **liveness** - Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live

**loudness** - Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude).

**speechiness** - Speechiness detects the presence of spoken words in a track.

**valence** - A measurement describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).

**tempo** - In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.

### Creating a Neural Network 

The neural network built for the Spotify Song Suggester worked, but unfortunately it was not scalable for the application's timeline.

It trained on the Kaggle-provided Spotify datasets that were pre-assigned/grouped into 400 clusters using sklearn.cluster's 'KMeans' model. 

In it's most efficient iteration, the neural network achieved a 96% accuracy/0.1038 loss predicting the KMeans cluster of each song in the training dataset. Accuracy and loss on the test dataset were 94.57% and 0.1667, respectively.

The Spotify data for our neural network was wrangled as follows:
- **speechiness** - limited dataset to tracks with ('speechiness' < 0.70).  This removed 1,000 rows from the dataset, and was done to reduce the impact of spoken word/introduction songs on the model.
- **popularity** - removed 'popularity' column from dataset entirely.  Popularity column is not a good feature to use for the neural network because it is constantly fluctuating.  For example, the popularity of the same songs changed in the time the 'Nov2018' and 'April2019' datasets were created.
- **duration_ms** - removed 'duration_ms' column from dataset entirely. Song length is a poor predictor of song similarity.  For example, we wouldn't want our neural network to think a rap song and an orchestral song belong in the same cluster because they are both 3 minutes, 30 seconds long.

Our neural network was built as follows:
- 'Sequential' model, from tensorflow.keras.models library
- 3 Dense layers:
- Layer 1: 12 nodes, 12 input dimensions (to match number of features fed to model), rectified linear unit activation function
- Layer 2: 512 nodes, sigmoid activation function
- Layer 3: 400 output dimensions (to match 400 KMeans clusters), softmax activation function. The softmax activation function in the output layer generates an array of probabilities for each song.  There are 400 probabilities in the array, which represent the likelihood a song belongs one of the 400 KMeans clusters.

Successes:
- Used RESTful API (described below) to pull song features for Elton John's 'Tiny Dancer', a piano-based rock song, and The Game's 'How We Do (featuring 50 Cent)', an uptempo string-based rap song.  
- Features were manually wrangled and fed to model, assigning both songs to a KMeans cluster
- KMeans cluster for 'Tiny Dancer' contained a lot of similar piano-based songs from the dataset!
- KMeans cluster for 'How We Do' contained a lot of similar rap songs from the dataset!

Failures:
- Input data had to be manually wrangled for input to model
- Model trained on only 400 KMeans clusters is not practical to scale to whole Spotify song library
- Dataset of 100,000 is not large enough to accurately predict similarities for a 30 million song library
- No way to scale model to entire Spotify library; could not assign KMeans cluster to every song, which is the only way the model could practically service our application
- model was not perfect; predicted clusters for both 'Tiny Dancer' and 'How We Do' contained unrelated songs


### RESTful API with Flask 

To power the Data Science aspect of this project, we needed to build a platform that received calls from the user application, collected data from the Spotify API, implemented our model, and then return rendered data back to the user application. 


![alt text](https://github.com/bw-spotify-oct/ds/blob/master/img/api.png "Logo Title Text 1")

#### Here are a few important calls we created for the API:
**Song search functionality** - Returns seven songs related to the user search query.

~~~
https://spotify-api-helper.herokuapp.com/songs/<api_key>/<user_query>
~~~

**Returns**
~~~
[
{"artist":"Florida Georgia Line","id":"498ZVInMGDkmmNVpSWqHiZ","song_name":"May We All","uri":"spotify:track:498ZVInMGDkmmNVpSWqHiZ"},
{"artist":"Florida Georgia Line","id":"58zWN3BNikOH7zVP6QGBZp","song_name":"May We All - Acoustic","uri":"spotify:track:58zWN3BNikOH7zVP6QGBZp"},
{"artist":"Florida Georgia Line","id":"6RHDliBPKS2TShFp7UIHF0","song_name":"May We All","uri":"spotify:track:6RHDliBPKS2TShFp7UIHF0"},
{"artist":"Florida Georgia Line","id":"6GIei0QWZjbrNWNwtTpiQL","song_name":"May We All - Commentary","uri":"spotify:track:6GIei0QWZjbrNWNwtTpiQL"},
{"artist":"Crooks UK","id":"37KWDhJ3fyBAFVcC0Hutan","song_name":"May Be","uri":"spotify:track:37KWDhJ3fyBAFVcC0Hutan"},
{"artist":"Florida Georgia Line","id":"0ONccS8Kchy2jgbhF5lg1i","song_name":"May We All","uri":"spotify:track:0ONccS8Kchy2jgbhF5lg1i"},
{"artist":"Roky Erickson","id":"6VbF4Hm5LTYgKH9R7ss2zZ","song_name":"We Are Never Talking","uri":"spotify:track:6VbF4Hm5LTYgKH9R7ss2zZ"}
]
~~~

**Graph URI & Recommended Song Data** - Returns the graph URI and five recommended songs based on a single song ID. 

~~~
https://spotify-api-helper.herokuapp.com/graph_data/<api_key>/<track_id>
~~~

**Returns**
~~~
[
{"graph_uri":"data:image/svg+xml;charset=utf-8;base64,PD94b...2c+PC9zdmc+"},
[{"artist":"Florida Georgia Line","id":"498ZVInMGDkmmNVpSWqHiZ","large_image":"https://i.scdn.co/image/205ec78bebfdc739bb3bb91076306e24f03c7d2c","med_image":"https://i.scdn.co/image/9ce932460729587fd8dccdac9214d741e127e162","small_image":"https://i.scdn.co/image/edc3a84866acc208acb5dbca21f05dda9a7eae99","song_name":"May We All","uri":"spotify:track:498ZVInMGDkmmNVpSWqHiZ"},
{"artist":"Maren Morris","id":"58spuRyMUsjKHQHEGwLC99","large_image":"https://i.scdn.co/image/ab67616d0000b2738bd4a20032a613d28e909109","med_image":"https://i.scdn.co/image/ab67616d00001e028bd4a20032a613d28e909109","small_image":"https://i.scdn.co/image/ab67616d000048518bd4a20032a613d28e909109","song_name":"80s Mercedes","uri":"spotify:track:58spuRyMUsjKHQHEGwLC99"},
{"artist":"Dustin Lynch","id":"2YMhrXQYKkm4kXLcXKKd5z","large_image":"https://i.scdn.co/image/ab67616d0000b2734686b4078ad5eb92315d28e0","med_image":"https://i.scdn.co/image/ab67616d00001e024686b4078ad5eb92315d28e0","small_image":"https://i.scdn.co/image/ab67616d000048514686b4078ad5eb92315d28e0","song_name":"Small Town Boy","uri":"spotify:track:2YMhrXQYKkm4kXLcXKKd5z"},
{"artist":"Maren Morris","id":"4H0vNUFcHPz5lytcLjwqkr","large_image":"https://i.scdn.co/image/ab67616d0000b2738bd4a20032a613d28e909109","med_image":"https://i.scdn.co/image/ab67616d00001e028bd4a20032a613d28e909109","small_image":"https://i.scdn.co/image/ab67616d000048518bd4a20032a613d28e909109","song_name":"Rich","uri":"spotify:track:4H0vNUFcHPz5lytcLjwqkr"},
{"artist":"Cole Swindell","id":"3y1t2sEahs8idFz2tiYNPO","large_image":"https://i.scdn.co/image/ab67616d0000b2734f239e3ce20b8e350244e84a","med_image":"https://i.scdn.co/image/ab67616d00001e024f239e3ce20b8e350244e84a","small_image":"https://i.scdn.co/image/ab67616d000048514f239e3ce20b8e350244e84a","song_name":"Let Me See Ya Girl","uri":"spotify:track:3y1t2sEahs8idFz2tiYNPO"},
{"artist":"Jon Pardi","id":"0bPnT6i9H1p8Vd85GS6Z7I","large_image":"https://i.scdn.co/image/ab67616d0000b2730504a96f206bc32ebe198b83","med_image":"https://i.scdn.co/image/ab67616d00001e020504a96f206bc32ebe198b83","small_image":"https://i.scdn.co/image/ab67616d000048510504a96f206bc32ebe198b83","song_name":"Night Shift","uri":"spotify:track:0bPnT6i9H1p8Vd85GS6Z7I"}]
]
~~~

**This is what the rendered URI of the chart looks like.**
![alt text](https://github.com/bw-spotify-oct/ds/blob/master/img/chart.png "Logo Title Text 1")

**Search Query with results** - Takes in a query and returns song search plus five recs. 

~~~
https://spotify-api-helper.herokuapp.com/auto_search/<api_key>/<query>
~~~

**Returns**
~~~
[
{"artist":"Florida Georgia Line","id":"498ZVInMGDkmmNVpSWqHiZ","song_name":"May We All","uri":"spotify:track:498ZVInMGDkmmNVpSWqHiZ"},
{"artist":"Maren Morris","id":"4H0vNUFcHPz5lytcLjwqkr","large_image":"https://i.scdn.co/image/ab67616d0000b2738bd4a20032a613d28e909109","med_image":"https://i.scdn.co/image/ab67616d00001e028bd4a20032a613d28e909109","small_image":"https://i.scdn.co/image/ab67616d000048518bd4a20032a613d28e909109","song_name":"Rich","uri":"spotify:track:4H0vNUFcHPz5lytcLjwqkr"},
{"artist":"Granger Smith","id":"11fWR3u9wjDMW4oVDbUbyT","large_image":"https://i.scdn.co/image/ab67616d0000b2737646f22eba3599fdeb6b7911","med_image":"https://i.scdn.co/image/ab67616d00001e027646f22eba3599fdeb6b7911","small_image":"https://i.scdn.co/image/ab67616d000048517646f22eba3599fdeb6b7911","song_name":"Happens Like That","uri":"spotify:track:11fWR3u9wjDMW4oVDbUbyT"},
{"artist":"Cole Swindell","id":"3y1t2sEahs8idFz2tiYNPO","large_image":"https://i.scdn.co/image/ab67616d0000b2734f239e3ce20b8e350244e84a","med_image":"https://i.scdn.co/image/ab67616d00001e024f239e3ce20b8e350244e84a","small_image":"https://i.scdn.co/image/ab67616d000048514f239e3ce20b8e350244e84a","song_name":"Let Me See Ya Girl","uri":"spotify:track:3y1t2sEahs8idFz2tiYNPO"},
{"artist":"Jon Pardi","id":"0bPnT6i9H1p8Vd85GS6Z7I","large_image":"https://i.scdn.co/image/ab67616d0000b2730504a96f206bc32ebe198b83","med_image":"https://i.scdn.co/image/ab67616d00001e020504a96f206bc32ebe198b83","small_image":"https://i.scdn.co/image/ab67616d000048510504a96f206bc32ebe198b83","song_name":"Night Shift","uri":"spotify:track:0bPnT6i9H1p8Vd85GS6Z7I"},
{"artist":"Frankie Ballard","id":"61jWRBNxdM3sO5oKLwZV9y","large_image":"https://i.scdn.co/image/ab67616d0000b27351f2b6933052de5d3c16fc7a","med_image":"https://i.scdn.co/image/ab67616d00001e0251f2b6933052de5d3c16fc7a","small_image":"https://i.scdn.co/image/ab67616d0000485151f2b6933052de5d3c16fc7a","song_name":"You'll Accomp'ny Me","uri":"spotify:track:61jWRBNxdM3sO5oKLwZV9y"}
]
~~~

**Returns Five Recs** - Takes ID and returns five recs. 

~~~
https://spotify-api-helper.herokuapp.com/recs/<api_key>/<track_id>
~~~

**Returns**
~~~
[
{"artist":"Florida Georgia Line","id":"498ZVInMGDkmmNVpSWqHiZ","song_name":"May We All","uri":"spotify:track:498ZVInMGDkmmNVpSWqHiZ"},
{"artist":"Maren Morris","id":"4H0vNUFcHPz5lytcLjwqkr","large_image":"https://i.scdn.co/image/ab67616d0000b2738bd4a20032a613d28e909109","med_image":"https://i.scdn.co/image/ab67616d00001e028bd4a20032a613d28e909109","small_image":"https://i.scdn.co/image/ab67616d000048518bd4a20032a613d28e909109","song_name":"Rich","uri":"spotify:track:4H0vNUFcHPz5lytcLjwqkr"},
{"artist":"Granger Smith","id":"11fWR3u9wjDMW4oVDbUbyT","large_image":"https://i.scdn.co/image/ab67616d0000b2737646f22eba3599fdeb6b7911","med_image":"https://i.scdn.co/image/ab67616d00001e027646f22eba3599fdeb6b7911","small_image":"https://i.scdn.co/image/ab67616d000048517646f22eba3599fdeb6b7911","song_name":"Happens Like That","uri":"spotify:track:11fWR3u9wjDMW4oVDbUbyT"},
{"artist":"Cole Swindell","id":"3y1t2sEahs8idFz2tiYNPO","large_image":"https://i.scdn.co/image/ab67616d0000b2734f239e3ce20b8e350244e84a","med_image":"https://i.scdn.co/image/ab67616d00001e024f239e3ce20b8e350244e84a","small_image":"https://i.scdn.co/image/ab67616d000048514f239e3ce20b8e350244e84a","song_name":"Let Me See Ya Girl","uri":"spotify:track:3y1t2sEahs8idFz2tiYNPO"},
{"artist":"Jon Pardi","id":"0bPnT6i9H1p8Vd85GS6Z7I","large_image":"https://i.scdn.co/image/ab67616d0000b2730504a96f206bc32ebe198b83","med_image":"https://i.scdn.co/image/ab67616d00001e020504a96f206bc32ebe198b83","small_image":"https://i.scdn.co/image/ab67616d000048510504a96f206bc32ebe198b83","song_name":"Night Shift","uri":"spotify:track:0bPnT6i9H1p8Vd85GS6Z7I"},
{"artist":"Frankie Ballard","id":"61jWRBNxdM3sO5oKLwZV9y","large_image":"https://i.scdn.co/image/ab67616d0000b27351f2b6933052de5d3c16fc7a","med_image":"https://i.scdn.co/image/ab67616d00001e0251f2b6933052de5d3c16fc7a","small_image":"https://i.scdn.co/image/ab67616d0000485151f2b6933052de5d3c16fc7a","song_name":"You'll Accomp'ny Me","uri":"spotify:track:61jWRBNxdM3sO5oKLwZV9y"}
]
~~~

**Mood Recs** - Takes mood values and saved songs. 

~~~
https://spotify-api-helper.herokuapp.com/playlist_mood_recs/<key>?acousticness=<num>&danceability=<num>&duration_ms=<num>&energy=<num>&instrumentalness=<num>&key=<num>&liveness=<num>&loudness=<num>&mode=<num>&speechiness=<num>&tempo=<num>&time_signature=<num>&valence=<num>&playlist=<[array or saved songs]>
~~~

**Returns**
~~~
[
{"graph_uri":"data:image/svg+xml;charset=utf-8;base64,PD94bWw...vc3ZnPg=="},{"acousticness":2.0,"danceability":1.33,"duration_ms":2.0,"energy":0.9,"instrumentalness":0.6,"key":0.9,"liveness":0.14,"loudness":0.7,"mode":1.0,"speechiness":0.09,"tempo":1.3,"time_signature":0.6,"valence":0.1},[{"artist":"Zac Brown Band","id":"4dGJf1SER1T6ooX46vwzRB","large_image":"https://i.scdn.co/image/ab67616d0000b2733f6f4ab78c05dfdfcc37a205","med_image":"https://i.scdn.co/image/ab67616d00001e023f6f4ab78c05dfdfcc37a205","small_image":"https://i.scdn.co/image/ab67616d000048513f6f4ab78c05dfdfcc37a205","song_name":"Chicken Fried","uri":"spotify:track:4dGJf1SER1T6ooX46vwzRB"},{"artist":"Trisha Yearwood","id":"2ulBBx6YQ3qY3ci34RadtN","large_image":"https://i.scdn.co/image/ab67616d0000b273bf4093011674adea23a1354b","med_image":"https://i.scdn.co/image/ab67616d00001e02bf4093011674adea23a1354b","small_image":"https://i.scdn.co/image/ab67616d00004851bf4093011674adea23a1354b","song_name":"She's In Love With The Boy - Single Version","uri":"spotify:track:2ulBBx6YQ3qY3ci34RadtN"},{"artist":"Russell Dickerson","id":"0vBMmF78QgtWY6dDNDPhbv","large_image":"https://i.scdn.co/image/ab67616d0000b273c1dfda1b94b3db5bcdeabf0d","med_image":"https://i.scdn.co/image/ab67616d00001e02c1dfda1b94b3db5bcdeabf0d","small_image":"https://i.scdn.co/image/ab67616d00004851c1dfda1b94b3db5bcdeabf0d","song_name":"Yours - Wedding Edition","uri":"spotify:track:0vBMmF78QgtWY6dDNDPhbv"},{"artist":"Dixie Chicks","id":"6cjwec9ii5uLK7CDfPBYt1","large_image":"https://i.scdn.co/image/ab67616d0000b2737c2a9c9ce2e9016223cdb74a","med_image":"https://i.scdn.co/image/ab67616d00001e027c2a9c9ce2e9016223cdb74a","small_image":"https://i.scdn.co/image/ab67616d000048517c2a9c9ce2e9016223cdb74a","song_name":"Wide Open Spaces","uri":"spotify:track:6cjwec9ii5uLK7CDfPBYt1"},{"artist":"Jerrod Niemann","id":"1HYKv0B6bYycqpxWHydKy9","large_image":"https://i.scdn.co/image/ab67616d0000b273c4391dcea9cbe6596f316e3c","med_image":"https://i.scdn.co/image/ab67616d00001e02c4391dcea9cbe6596f316e3c","small_image":"https://i.scdn.co/image/ab67616d00004851c4391dcea9cbe6596f316e3c","song_name":"One More Drinkin' Song","uri":"spotify:track:1HYKv0B6bYycqpxWHydKy9"}]
]
~~~

**Playlist Recs** - Takes saved songs and returns recs. 

~~~
https://spotify-api-helper.herokuapp.com/playlist_recs/<key>?playlist=<[array or saved songs]>
~~~

**Returns**
~~~
[
{"graph_uri":"data:image/svg+xml;charset=utf-8;base64,PD94bWw...vc3ZnPg=="},{"acousticness":2.0,"danceability":1.33,"duration_ms":2.0,"energy":0.9,"instrumentalness":0.6,"key":0.9,"liveness":0.14,"loudness":0.7,"mode":1.0,"speechiness":0.09,"tempo":1.3,"time_signature":0.6,"valence":0.1},[{"artist":"Zac Brown Band","id":"4dGJf1SER1T6ooX46vwzRB","large_image":"https://i.scdn.co/image/ab67616d0000b2733f6f4ab78c05dfdfcc37a205","med_image":"https://i.scdn.co/image/ab67616d00001e023f6f4ab78c05dfdfcc37a205","small_image":"https://i.scdn.co/image/ab67616d000048513f6f4ab78c05dfdfcc37a205","song_name":"Chicken Fried","uri":"spotify:track:4dGJf1SER1T6ooX46vwzRB"},{"artist":"Trisha Yearwood","id":"2ulBBx6YQ3qY3ci34RadtN","large_image":"https://i.scdn.co/image/ab67616d0000b273bf4093011674adea23a1354b","med_image":"https://i.scdn.co/image/ab67616d00001e02bf4093011674adea23a1354b","small_image":"https://i.scdn.co/image/ab67616d00004851bf4093011674adea23a1354b","song_name":"She's In Love With The Boy - Single Version","uri":"spotify:track:2ulBBx6YQ3qY3ci34RadtN"},{"artist":"Russell Dickerson","id":"0vBMmF78QgtWY6dDNDPhbv","large_image":"https://i.scdn.co/image/ab67616d0000b273c1dfda1b94b3db5bcdeabf0d","med_image":"https://i.scdn.co/image/ab67616d00001e02c1dfda1b94b3db5bcdeabf0d","small_image":"https://i.scdn.co/image/ab67616d00004851c1dfda1b94b3db5bcdeabf0d","song_name":"Yours - Wedding Edition","uri":"spotify:track:0vBMmF78QgtWY6dDNDPhbv"},{"artist":"Dixie Chicks","id":"6cjwec9ii5uLK7CDfPBYt1","large_image":"https://i.scdn.co/image/ab67616d0000b2737c2a9c9ce2e9016223cdb74a","med_image":"https://i.scdn.co/image/ab67616d00001e027c2a9c9ce2e9016223cdb74a","small_image":"https://i.scdn.co/image/ab67616d000048517c2a9c9ce2e9016223cdb74a","song_name":"Wide Open Spaces","uri":"spotify:track:6cjwec9ii5uLK7CDfPBYt1"},{"artist":"Jerrod Niemann","id":"1HYKv0B6bYycqpxWHydKy9","large_image":"https://i.scdn.co/image/ab67616d0000b273c4391dcea9cbe6596f316e3c","med_image":"https://i.scdn.co/image/ab67616d00001e02c4391dcea9cbe6596f316e3c","small_image":"https://i.scdn.co/image/ab67616d00004851c4391dcea9cbe6596f316e3c","song_name":"One More Drinkin' Song","uri":"spotify:track:1HYKv0B6bYycqpxWHydKy9"}]
]
~~~


**Song search with images** - Takes a query and returns related songs along with pics. 

~~~
https://spotify-api-helper.herokuapp.com/songs_with_pic/<key>/<query>
~~~

**Returns**
~~~
[
{"artist":"Florida Georgia Line","id":"498ZVInMGDkmmNVpSWqHiZ","large_image":"https://i.scdn.co/image/205ec78bebfdc739bb3bb91076306e24f03c7d2c","med_image":"https://i.scdn.co/image/9ce932460729587fd8dccdac9214d741e127e162","small_image":"https://i.scdn.co/image/edc3a84866acc208acb5dbca21f05dda9a7eae99","song_name":"May We All","uri":"spotify:track:498ZVInMGDkmmNVpSWqHiZ"},
{"artist":"Florida Georgia Line","id":"58zWN3BNikOH7zVP6QGBZp","large_image":"https://i.scdn.co/image/3a521ef461bba97d0d4e0a748bbcbc5c9e70b81b","med_image":"https://i.scdn.co/image/7f8fcf96edf847f839ad4a825c304e6e744fe737","small_image":"https://i.scdn.co/image/e6906b22f466cba7cc4544b5e89516fd511770c1","song_name":"May We All - Acoustic","uri":"spotify:track:58zWN3BNikOH7zVP6QGBZp"},
{"artist":"Florida Georgia Line","id":"6RHDliBPKS2TShFp7UIHF0","large_image":"https://i.scdn.co/image/66594e26e0a43b69b8ab6b78df80b0b3c831c5fa","med_image":"https://i.scdn.co/image/39260082f7ac1d7a79f28f0aa2bb5f11bc8279a2","small_image":"https://i.scdn.co/image/5085916175ee37e0582c55a28dfec8759d2bd7b4","song_name":"May We All","uri":"spotify:track:6RHDliBPKS2TShFp7UIHF0"},
{"artist":"Florida Georgia Line","id":"6GIei0QWZjbrNWNwtTpiQL","large_image":"https://i.scdn.co/image/66594e26e0a43b69b8ab6b78df80b0b3c831c5fa","med_image":"https://i.scdn.co/image/39260082f7ac1d7a79f28f0aa2bb5f11bc8279a2","small_image":"https://i.scdn.co/image/5085916175ee37e0582c55a28dfec8759d2bd7b4","song_name":"May We All - Commentary","uri":"spotify:track:6GIei0QWZjbrNWNwtTpiQL"},
{"artist":"Crooks UK","id":"37KWDhJ3fyBAFVcC0Hutan","large_image":"https://i.scdn.co/image/ab67616d0000b2736dd78f14c86f1d2d183db98c","med_image":"https://i.scdn.co/image/ab67616d00001e026dd78f14c86f1d2d183db98c","small_image":"https://i.scdn.co/image/ab67616d000048516dd78f14c86f1d2d183db98c","song_name":"May Be","uri":"spotify:track:37KWDhJ3fyBAFVcC0Hutan"},
{"artist":"Florida Georgia Line","id":"0ONccS8Kchy2jgbhF5lg1i","large_image":"https://i.scdn.co/image/ab67616d0000b27385bdb09f395d298a77c2636b","med_image":"https://i.scdn.co/image/ab67616d00001e0285bdb09f395d298a77c2636b","small_image":"https://i.scdn.co/image/ab67616d0000485185bdb09f395d298a77c2636b","song_name":"May We All","uri":"spotify:track:0ONccS8Kchy2jgbhF5lg1i"},
{"artist":"Roky Erickson","id":"6VbF4Hm5LTYgKH9R7ss2zZ","large_image":"https://i.scdn.co/image/ab67616d0000b273790213e5c993dc263f735e69","med_image":"https://i.scdn.co/image/ab67616d00001e02790213e5c993dc263f735e69","small_image":"https://i.scdn.co/image/ab67616d00004851790213e5c993dc263f735e69","song_name":"We Are Never Talking","uri":"spotify:track:6VbF4Hm5LTYgKH9R7ss2zZ"}
]
~~~
