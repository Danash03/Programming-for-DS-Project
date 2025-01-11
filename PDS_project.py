import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, normalize, Binarizer, StandardScaler, LabelEncoder
import multiprocessing
import time 
import requests
from pyquery import PyQuery as pq
import json

#----------Pandas-------------------------
spotify = pd.read_csv("C:/Users/Mohammad/Desktop/Programming for DS Project/myenv/Spotify Most Streamed Songs.csv")
print(spotify.describe())
print(spotify.info())
print(spotify.columns)
print(spotify.dtypes)
print(spotify.head(2))
print(spotify.tail(2))
filtered_songs = spotify[(spotify['artist(s)_name'] == 'The Weeknd') | (spotify['artist(s)_name'] == 'Taylor Swift')]
print(filtered_songs)
print(spotify)
tracks_2023 = spotify[spotify['released_year'] == 2023]
print(tracks_2023[['track_name', 'released_year']])
del spotify['in_deezer_playlists']
del spotify['in_deezer_charts']
streams_sorted = spotify.sort_values(by='streams', ascending=False)
print(streams_sorted)
data = {
    'track_name': ['Blinding Lights', 'Watermelon Sugar', 'Levitating'],
    'artist': ['The Weeknd', 'Harry Styles', 'Dua Lipa'],
    'streams': [2600000000, 1500000000, 1800000000]
}
spotify_df = pd.DataFrame(data)
print(spotify_df)
print(spotify['track_name'])  # Returns a Series
print(spotify[['track_name', 'streams']])  # Returns a DataFrame
print(spotify['streams'].dtype)
print(spotify['streams'].unique())
print(spotify.isnull().sum())
d_shazam = spotify['in_shazam_charts'].dropna(axis=0) #axis=1
print(d_shazam)
spotify.fillna({'key':'not available'},inplace=True)#make better
print(spotify.isnull().sum())#doesn't have null values (95)anymore
for index, row in spotify.iterrows():
    print(f"Track: {row['track_name']} | Streams: {row['streams']}")
spotify['stream_rank'] = spotify['streams'].rank(ascending=False)
print(spotify[['track_name', 'streams', 'stream_rank']].head())
df_2022 = spotify[spotify['released_year'] == 2022]
df_2023 = spotify[spotify['released_year'] == 2023]
combined_df = pd.concat([df_2022, df_2023])
print(combined_df)
print(spotify[['released_year', 'released_month', 'released_day']].dtypes)#we wanted to check their data types before converting

spotify['release_date'] = pd.to_datetime(
    [
        f"{year}-{str(month).zfill(2)}-{str(day).zfill(2)}"
        for year, month, day in zip(spotify['released_year'], spotify['released_month'], spotify['released_day'])
    ],
    format='%Y-%m-%d', errors='coerce'
)

print(spotify['release_date'])

harry_styles_songs = spotify[spotify['artist(s)_name'].str.lower() == 'harry styles']

sorted_songs = harry_styles_songs.sort_values(by='streams')

most_streamed_release_date = sorted_songs.iloc[-1]['release_date']
least_streamed_release_date = sorted_songs.iloc[0]['release_date']

time_difference = most_streamed_release_date - least_streamed_release_date
print(time_difference)

spotify['released_year'] = pd.Categorical(spotify['released_year'])
songs_per_year = spotify['released_year'].value_counts().sort_index()
print(songs_per_year)
spotify['streams'] = pd.to_numeric(spotify['streams'], errors='coerce')

spotify = spotify.dropna(subset=['streams'])

#---------scikit-learn-------------------------
# Binarization
binarizer = Binarizer(threshold=1_000_000_000)
spotify['Binarized_Streams'] = binarizer.fit_transform(spotify[['streams']])
print("Binarized Streams:")
print(spotify[['streams', 'Binarized_Streams']].head())

# Label coding for 'mode'
encoder = LabelEncoder()
spotify['mode'] = encoder.fit_transform(spotify['mode'])
print("\nLabel Encoded Mode:")
print(spotify[['mode']].head())

# Normalization
spotify[['streams', 'in_spotify_playlists']] = normalize(spotify[['streams', 'in_spotify_playlists']], axis=0)
print("\nNormalized Streams and In Spotify Playlists:")
print(spotify[['streams', 'in_spotify_playlists']].head())

# Standardization
standard_scaler = StandardScaler()
spotify[['bpm', 'streams']] = standard_scaler.fit_transform(spotify[['bpm', 'streams']])
print("\nStandardized BPM and Streams:")
print(spotify[['bpm', 'streams']].head())

# Scaling using MinMaxScaler
scaler = MinMaxScaler()
scaled_columns = ['bpm', 'danceability_%', 'energy_%', 'acousticness_%', 'valence_%']
spotify[scaled_columns] = scaler.fit_transform(spotify[scaled_columns])
print("\nMinMax Scaled Columns:")
print(spotify[scaled_columns].head())

# Check the result
print("\nFinal Processed Data:")
print(spotify.head())
#----------JSON------------
#spotify = pd.read_csv("Spotify Most Streamed Songs.csv")
spotify.to_json("Spotify_Most_Streamed_Songs.json", index=False)
print("CSV data has been converted to JSON format and saved.")
# Subset: Songs by The Weeknd
lana_del_ray_songs = spotify[spotify['artist(s)_name'].str.contains('Lana Del Rey')]

# Write to JSON
lana_del_ray_songs.to_json('lana_del_ray_songs.json', index=False)
# Extract top 10 songs by streams
spotify['streams'] = pd.to_numeric(spotify['streams'], errors='coerce')

top_10_songs = spotify.nlargest(10, 'streams')

# Write this to a new JSON file
top_10_songs.to_json('Top_10_Songs.json', index=False)

# Get unique artists
unique_artists = spotify['artist(s)_name'].unique()

# Number of unique artists
num_unique_artists = len(unique_artists)

print(f"Number of unique artists: {num_unique_artists}")
print("Sample of unique artists:", unique_artists[:10])
# Create a subset with release year and track name
release_track_dict = spotify[['released_year', 'track_name']].to_dict(orient='records')

# Convert to JSON string
release_track_json = json.dumps(release_track_dict, indent=4)

# Print the JSON string
print(release_track_json)
# Save the release year and track names JSON data to a file
with open('Release_Track_Data.json', 'w') as json_file:
    json.dump(release_track_dict, json_file, indent=4)
# Load the JSON file into a Python object (list of dictionaries)
with open('Top_10_Songs.json', 'r') as json_file:
    loaded_songs_list = json.load(json_file)
print(loaded_songs_list)
# Read the JSON file contents into a string
with open('lana_del_ray_songs.json', 'r') as json_file:
    json_string = json_file.read()

# Load the JSON string into a Python object (list of dictionaries)
loaded_lana_del_ray_songs = json.loads(json_string)
print(loaded_lana_del_ray_songs)
released_tracks=pd.read_json('Release_Track_Data.json')
print(released_tracks)
print(released_tracks.info())
#---------MultiThreading-------------
# Function to calculate total streams for a specific playlist column
def calculate_streams(playlist_name, data):
    print(f"Processing {playlist_name}...")
    total_streams = data[playlist_name].sum()
    time.sleep(2)  
    print(f"Total streams for {playlist_name}: {total_streams}")

if __name__ == "__main__":
    playlist_columns = ['in_spotify_playlists', 'in_apple_playlists']


    processes = []
    for playlist in playlist_columns:
        process = multiprocessing.Process(target=calculate_streams, args=(playlist, spotify))
        processes.append(process)
        process.start()
    for process in processes:
        process.join()
        
        print("done!")
#  a function to calculate danceability for each song by harry styles
def calculate_danceability_for_song(song):
    song_name = song['track_name']
    danceability = song['danceability_%']
    time.sleep(0.5) 
    print(f"Song: {song_name}, Danceability: {danceability:.2f}%")  
    return song_name, danceability

if __name__ == "__main__":
    harry_styles_songs = spotify[spotify['artist(s)_name'] == 'Harry Styles']

    pool = multiprocessing.Pool(processes=2)
    results = pool.map(calculate_danceability_for_song, harry_styles_songs.to_dict('records'))

    pool.close()
    pool.join()

    print("\nAll songs processed.")


#  a function that processes the streams for each song by Post Malone, Justin Bieber
def process_streams_for_song(song, queue):
    song_name = song['track_name']
    streams = song['streams']
    time.sleep(1)  
    print(f"Processing {song_name}...")
    queue.put((song_name, streams))  

if __name__ == "__main__":
    artists_of_interest = ['Post Malone', 'Justin Bieber']
    filtered_songs = spotify[spotify['artist(s)_name'].isin(artists_of_interest)]

    comm_queue = multiprocessing.Queue()
    workers = []

    for artist in artists_of_interest:
        artist_songs = filtered_songs[filtered_songs['artist(s)_name'] == artist]

        print(f"Processing songs by {artist}...\n")

        for _, song in artist_songs.iterrows():
            worker = multiprocessing.Process(target=process_streams_for_song, args=(song, comm_queue))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

    results = []
    while not comm_queue.empty():
        result = comm_queue.get()
        results.append(result)

    for song_name, streams in results:
        print(f"Song: {song_name}, Streams: {streams}")

    print("Done!")
#____________webscarping&crawling___________________
def simple_scrape_lyrics_url(song_title, artist_name):
    search_url = f"https://genius.com/api/search/multi?q={song_title} {artist_name}"
    response = requests.get(search_url).json()

    if response['response']['sections']:
        song_url = response['response']['sections'][0]['hits'][0]['result']['url']
        return f"Lyrics available at: {song_url}"

    return "Lyrics not found"

print(simple_scrape_lyrics_url("Glimpse of Us", "Joji"))

def fetch_json_data(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()  # Parse and return JSON data
    else:
        return f"Failed to fetch data. Status code: {response.status_code}"

# Example usage (using a placeholder URL)
example_url = "https://api.spotify.com/v1/tracks/{track_id}"
json_data = fetch_json_data(example_url)
print(json_data)
import requests

def post_json_data(url, data):
    headers = {'Content-Type': 'application/json'}
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        return response.json()  # Return response as JSON
    else:
        return f"Failed to post data. Status code: {response.status_code}"

# Example usage
example_url = "https://api.spotify.com/v1/users/{user_id}/playlists"
data_to_send = {"key1": "value1", "key2": "value2"}
response_data = post_json_data(example_url, data_to_send)
print(response_data)

url = "https://jsonplaceholder.typicode.com/posts"  # A mock API endpoint for demonstration

# Parameters related to the dataset (using song information as an example)
data = {
    "track_name": "Shape of You",
    "artist_name": "Ed Sheeran",
    "streams": 3000000000,
    "released_year": 2017
}

# Send POST request with parameters (data)
post_result = requests.post(url, json=data)

# Check if the request was successful
if post_result.status_code == 201:  # 201 indicates successful creation
    print(post_result.json())  # Print the response JSON
else:
    print("Error: Unable to post data")
    

# Load the page
doc = pq(url="https://en.wikipedia.org/wiki/Spotify")

# Count the number of <table> elements
print(f"Number of tables on the page: {len(doc('table'))}")

# Filter the wikitable class
wikitable = doc('.wikitable')
print(f"Number of tables with the class 'wikitable': {len(wikitable)}")

# Loop through each row in the table
for row in wikitable('tr'):
    td = pq(row)  # Parse each row into a PyQuery object
    # Get and print the text of each table cell (td)
    print(td.text())