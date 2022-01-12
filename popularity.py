import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import datetime

from pyspark.sql import SparkSession, functions, types

spark = SparkSession.builder.appName('reddit averages').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
assert spark.version >= '2.3' # make sure we have Spark 2.3+




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


def main(in_directory):
    songs = spark.read.csv(in_directory, schema=songs_schema, header=True)

    # drop columns with null values
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
    
    songs = songs.drop(
        'release_date',
        'name', 
        'key', 
        'id', 
        'artists',
        'explicit',
    )

    # convert to Pandas to train model
    songs_pd = songs.toPandas()
    

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
    X = mmscaler.fit_transform(songs_pd.drop('popularity', 1))
    y = songs_pd['popularity']
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)

    # train model
    # best result obtained with min_samples = 200, max_depth = 15
    model = RandomForestRegressor(100, max_depth=15)
    model.fit(X_train, y_train)
    print(model.score(X_valid, y_valid))
   

if __name__=='__main__':
    in_directory = sys.argv[1]
    main(in_directory)
