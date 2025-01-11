import pandas as pd
from fastapi import FastAPI

app = FastAPI()

# Load the CSV file
#data = pd.read_csv("spotify_most_streamed_songs.csv")

data = pd.read_csv("C:/Users/Mohammad/Desktop/Programming for DS Project/myenv/Spotify Most Streamed Songs.csv")


@app.get("/")
def read_root():
    return {"message": "Welcome to the Spotify Most Streamed Songs API!"}

@app.get("/tracks/{artist_name}")
def get_tracks_by_artist(artist_name: str):
    # Filter the data for the given artist name (case-insensitive)
    filtered_data = data[data["artist(s)_name"].str.lower() == artist_name.lower()]
    
    # Extract only the track names and return as a list
    track_names = filtered_data["track_name"].tolist()
    return {"track_names": track_names}


@app.get("/year/{year}")
def get_tracks_by_year(year: int):
    # Filter the data for the given released year
    filtered_data = data[data["released_year"] == year]
    track_names = filtered_data["track_name"].tolist()
    return {"track_names": track_names}