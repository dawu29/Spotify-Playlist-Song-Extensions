**<h1>Spotify Data Exploration</h1>**

**<h2>Popularity Prediction Model</h2>**

The code that was used to gain insight into what makes a song popular and to 
train a model that predicts popularity can be found in `popularity.ipynb`.
Running the code in this file should be trivial.

The code in `popularity.py` iterates on this first model with the use of a
Random Forest Regressor.

*<h2>To run the code:</h2>*

Run spark locally

```
PYSPARK_PYTHON=python3
export PATH=${PATH}:{your path to spark/bin}
```

Then:

```
spark-submit popularity.py spotify-songs.csv
```

<br>

**<h2>Playlist Song Extension</h2>**

The code that generates song recommendations from playlists is `playlist_recommend.py`.

The data that `playlist_recommend.py` uses can be found in 
`spotify-songs.csv` and `spotify-playlists.csv`


*<h2>To run the code:</h2>*

Run spark locally

```
PYSPARK_PYTHON=python3
export PATH=${PATH}:{your path to spark/bin}
```

Then:

```
spark-submit playlist_recommend.py spotify-playlists.csv spotify-data.csv output.csv
```


