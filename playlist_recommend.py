import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from pyspark.sql import SparkSession, functions, types, Row

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+


playlists_schema = types.StructType([
    types.StructField('user_id', types.StringType()),
    types.StructField('artist', types.StringType()),
    types.StructField('song', types.StringType()),
    types.StructField('playlist', types.StringType()),
])

songs_schema = types.StructType([
    types.StructField('valence', types.FloatType()),
    types.StructField('year', types.DateType()),
    types.StructField('acousticness', types.FloatType()),
    types.StructField('artists', types.StringType()),
    types.StructField('danceability', types.FloatType()),
    types.StructField('duration_ms', types.LongType()),
    types.StructField('energy', types.FloatType()),
    types.StructField('explicit', types.BooleanType()),
    types.StructField('id', types.StringType()),
    types.StructField('instrumentalness', types.FloatType()),
    types.StructField('key', types.IntegerType()),
    types.StructField('liveness', types.FloatType()),
    types.StructField('loudness', types.FloatType()),
    types.StructField('mode', types.IntegerType()),
    types.StructField('name', types.StringType()),
    types.StructField('popularity', types.IntegerType()),
    types.StructField('release_date', types.DateType()),
    types.StructField('speechiness', types.FloatType()),
    types.StructField('tempo', types.FloatType()),
])


def main(playlist_directory, songs_directory, out_directory):
    playlists = spark.read.csv(playlist_directory, schema=playlists_schema, header=False)
    songs = spark.read.csv(songs_directory, schema=songs_schema, header=True)

    # drop columns with null values - playlists
    playlists = playlists.dropna()

    # drop columns with null values - songs
    songs = songs.dropna(
        subset=(
            'valence',
            'year',
            'acousticness',
            'danceability',
            'duration_ms',
            'energy',
            'id',
            'instrumentalness',
            'key',
            'liveness',
            'loudness',
            'mode',
            'name',
            'popularity',
            'release_date',
            'speechiness',
            'tempo'
        )
    )

    # convert ms to minutes
    songs = songs \
        .withColumn('duration', (songs['duration_ms'] / 60000).cast(types.FloatType())) \
        .drop('duration_ms')

    # drop songs longer than 12 minutes
    songs = songs.filter((songs['duration'] <= 12))

    # transform data
    songs = songs \
        .withColumn('speechiness', functions.log10(songs['speechiness']+0.001)) \
        .withColumn('acousticness', functions.sqrt(songs['acousticness']+0.001)) \
        .withColumn('liveness', functions.log10(songs['liveness']+0.001)) \
        .withColumn('instrumentalness', functions.sqrt(songs['instrumentalness']+0.001))

    # convert to Pandas to train model
    songs_pd = songs.drop(
        'release_date',
        'key', 
        'id', 
        'artists',
        'explicit',
    ).toPandas()

    # adjust year
    songs_pd['year'] = songs_pd['year'].apply(lambda d: d.strftime('%Y'))

    # scale values
    mmscaler = MinMaxScaler()
    scaled = mmscaler.fit_transform(
        songs_pd[[
            'acousticness',
            'liveness',
            'loudness',
            'speechiness',
            'tempo',
            'duration',
            'instrumentalness'
        ]]
    )

    songs_pd[[
        'acousticness',
        'liveness',
        'loudness',
        'speechiness',
        'tempo',
        'duration',
        'instrumentalness'
    ]] = scaled

    # split data
    X = mmscaler.fit_transform(songs_pd.drop(['popularity', 'name'], 1))
    y = songs_pd['popularity']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # train model
    print('Training Song Popularity Prediction Model')
    model = RandomForestRegressor(50, max_depth=15)
    model.fit(X_train, y_train)
    print("Evaluating Song Popularity Prediction Model")
    print(model.score(X_valid, y_valid))

    # convert df back to spark to compute playlist feature averages
    songs = spark.createDataFrame(songs_pd)

    # join songs and playlists
    joined = playlists.join(
        songs,
        playlists['song'] == songs['name']
    ).drop(
        'user_id',
        'release_date',
        'id',
        'artists'
    )

    # find average for each playlist
    print("Calculating average feature values for each playlist")
    average_feature_by_playlist = joined.groupBy('playlist').agg(
        functions.avg('valence'),
        functions.avg('year'),
        functions.avg('acousticness'),
        functions.avg('danceability'),
        functions.avg('energy'),
        functions.avg('instrumentalness'),
        functions.avg('liveness'),
        functions.avg('loudness'),
        functions.avg('speechiness'),
        functions.avg('tempo'),
        functions.avg('duration'),
        functions.avg('mode')
    ).drop(
        'song',
        'artist',
        'mode',
        'name',
        'popularity'
    )

    # convert to pandas
    average_feature_by_playlist_sample = average_feature_by_playlist.sample(.05, 3)
    average_feature_by_playlist_pd = average_feature_by_playlist_sample.toPandas()

    # calculate euclidean distance between the playlist and every song
    def get_recommendations(playlist):
        print('     for: ', playlist['playlist'])
        playlist = pd.DataFrame([{
                'playlist': playlist['playlist'],
                'avg(valence)': playlist['avg(valence)'],
                'avg(year)': playlist['avg(year)'],
                'avg(acousticness)': playlist['avg(acousticness)'],
                'avg(danceability)': playlist['avg(danceability)'],
                'avg(energy)': playlist['avg(energy)'],
                'avg(instrumentalness)': playlist['avg(instrumentalness)'],
                'avg(liveness)': playlist['avg(liveness)'],
                'avg(loudness)': playlist['avg(loudness)'],
                'avg(speechiness)': playlist['avg(speechiness)'],
                'avg(tempo)': playlist['avg(tempo)'],
                'avg(duration)': playlist['avg(duration)'],
                'avg(mode)': playlist['avg(mode)']
        }])

        # Calculate euclidean distance
        songs_dist = songs_pd
        songs_dist['distance'] = np.linalg.norm(
            playlist[[
                'avg(valence)',
                'avg(acousticness)',
                'avg(danceability)',
                'avg(energy)',
                'avg(instrumentalness)',
                'avg(liveness)',
                'avg(loudness)',
                'avg(speechiness)',
                'avg(tempo)',
                'avg(duration)',
                'avg(mode)'
            ]].values - songs_pd[[
                'valence',
                'acousticness',
                'danceability',
                'energy',
                'instrumentalness',
                'liveness',
                'loudness',
                'speechiness',
                'tempo',
                'duration',
                'mode'
            ]].values,
            axis=1
        )

        # Calculate popularity using model
        songs_dist = songs_dist.sort_values(by=['distance'], ascending=True).head(5)
        songs_dist['_popularity'] = model.predict(
            songs_dist[[
                'valence',
                'year',
                'acousticness',
                'danceability',
                'energy',
                'instrumentalness',
                'liveness',
                'loudness',
                'mode',
                'speechiness',
                'tempo',
                'duration',
            ]].values
        )

        # Calculate likelihood score
        songs_dist['_likelihood'] = \
            songs_dist['_popularity'] * (np.power(songs_dist['distance'],-1))
        songs_dist = songs_dist.sort_values(by=['_likelihood'], ascending=False)
        songs_dist['playlist'] = playlist['playlist'].iloc[0]

        return \
            songs_dist['playlist'].values[0], \
            songs_dist['name'].values, \
            songs_dist['_likelihood'].values, \
            songs_dist['_popularity'].values, \
            songs_dist['distance'].values

    # Generate 5 song recommendations for each playlist
    print('Generating recommendations')
    playlist_extension_pd = pd.DataFrame(
        columns=['playlist', 'name', '_likelihood', '_popularity', 'distance']
    )
    playlist_extension_pd['playlist'], \
    playlist_extension_pd['name'], \
    playlist_extension_pd['_likelihood'], \
    playlist_extension_pd['_popularity'], \
    playlist_extension_pd['distance'] = zip(
        *average_feature_by_playlist_pd.apply(get_recommendations, axis=1)
    )

    # # visualization
    # pd.set_option('display.max_columns', None)
    # pd.set_option('max_colwidth', 800)
    # print(playlist_extension_pd)

    # save result
    # playlist_extension_pd.to_csv(out_directory)



if __name__=='__main__':
    playlist_directory = sys.argv[1]
    song_directory = sys.argv[2]
    out_directory = sys.argv[3]
    main(playlist_directory, song_directory, out_directory)
