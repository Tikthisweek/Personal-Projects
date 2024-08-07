#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 16:08:27 2024

@author: KartikPatel
"""

import requests
import re
import webbrowser

def main():
    while True:
        # Prompt user to choose default auth details or enter their own
        answer = input("Press 1 if you would like to use default auth details or 2 to enter your own: ")
        if re.search("^[1-2]$", answer):
            break
        else:
            print("Please enter valid value")

    if answer == "1":
        # Get auth token using default auth details
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_data = default_auth()
        auth_response = requests.post(auth_url, data = auth_data)
        # Extract token and call API
        if auth_response.status_code == 200:
            auth_token = auth_response.json()['access_token']
            entry = user_selection()
            headers = {
                'Authorization': f'Bearer {auth_token}'
            }
    else:
        # Get auth token using user-entered auth details
        auth_url = 'https://accounts.spotify.com/api/token'
        auth_data = enter_auth_details()
        auth_response = requests.post(auth_url, data = auth_data)
        if auth_response.status_code == 200:
            auth_token = auth_response.json()['access_token']
            entry = user_selection()
            headers = {
                'Authorization': f'Bearer {auth_token}'
            }
    # top tracks
    if entry == 'top tracks':
        artist_name = get_artist_name()
        while True:
            # Prompt user for number of tracks to be returned
            limit = input("Enter how many tracks would you like returned? ")
            try:
                limit = int(limit)
                break
            except ValueError:
                print("Please enter digit")

        # Search for top tracks by artist name
        params = {"q": artist_name, "type": 'track', "limit": limit}
        search_response = requests.get('https://api.spotify.com/v1/search', headers = headers, params = params)
        if search_response.status_code == 200:
            search_info = search_response.json()
            for i in search_info['tracks']['items']:
                print(i['name'])
            print(f"Artist Popularity: {search_info['tracks']['items'][0]['popularity']}")

            # Option to compare popularity with another artist
            follow_up = input("Would you like to compare popularity to another artist? Please enter Yes or No: ").lower()
            if re.search("^yes$", follow_up):
                first_artist_popularity = int(search_info['tracks']['items'][0]['popularity'])
                second_artist = get_artist_name()
                second_params = {"q": second_artist, "type": 'track'}
                second_search_response = requests.get('https://api.spotify.com/v1/search', headers = headers, params = second_params)
                second_search_info = second_search_response.json()
                second_artist_popularity = int(second_search_info['tracks']['items'][0]['popularity'])

                # Compare popularity
                if first_artist_popularity > second_artist_popularity:
                    print(f"{artist_name.title()}({first_artist_popularity}) is more popular than {second_artist.title()}({second_artist_popularity})")
                elif second_artist_popularity > first_artist_popularity:
                    print(f"{second_artist.title()}({second_artist_popularity}) is more popular than {artist_name.title()}({first_artist_popularity})")
                else:
                    print("Both are artists are of equal popularity")
            else:
                print("Have a good day!")
        else:
            print("Invalid URL")
    # display song
    elif entry == 'display song':
        artist_name = get_artist_name()
        song_name = input("Enter song name: ")
        params = {"q": artist_name, "type": 'track'}
        search_response = requests.get('https://api.spotify.com/v1/search', headers = headers, params = params)
        if search_response.status_code == 200:
            search_info = search_response.json()
            for i in search_info['tracks']['items']:
                if re.search(song_name, i['name'], re.IGNORECASE):
                    webbrowser.open(f'http://open.spotify.com/track/{i["id"]}')
        else:
            print("Invalid URL")
    # playlist
    elif entry == 'playlist':
        user_id = input("Enter user ID: ")
        while True:
            # Prompt user for number of playlists to be returned
            limit = input("Enter how many playlists would you like returned? ")
            try:
                limit = int(limit)
                break
            except ValueError:
                print("Please enter digit")

        # Search for playlists by user ID
        params = {"user_id": user_id, "limit": limit}
        search_response = requests.get(f'https://api.spotify.com/v1/users/{user_id}/playlists', headers = headers, params = params)
        if search_response.status_code == 200:
            search_info = search_response.json()
            playlist_list = []
            for i in search_info['items']:
                playlist_list.append(i['name'])
                print(i['name'])

            # Option to display a specific playlist
            answer = input("Enter Yes if you would like to display a playlist ").lower()
            if answer == "yes":
                print(playlist_list)
                playlist_name = input("Enter playlist name from above list ")
                for i in search_info['items']:
                    if re.search(playlist_name, i['name'], re.IGNORECASE):
                        webbrowser.open(f'http://open.spotify.com/playlist/{i["id"]}')
            else:
                print("Have a good day!")

def default_auth():
    # Return default auth details
    data = {
        'grant_type': 'client_credentials',
        'client_id': 'bc7d93ab04a74dfc92993c4ba8a0d8ac',
        'client_secret': '59bfdcd8c5ce4136b6df0c64f1b75980'
    }
    return data

def enter_auth_details():
    while True:
        # Prompt user for client ID
        input_id = input("Enter your client ID ").strip()
        if re.search(r'^[0-9a-zA-Z]{32}$', input_id):
            break
        else:
            print("Enter valid id")
    while True:
        # Prompt user for client Secret
        input_secret = input("Enter your client Secret ").strip()
        if re.search(r'^[0-9a-zA-Z]{32}$', input_secret):
            break
        else:
            print("Enter valid secret")

    # Return user-entered auth details
    data = {
        'grant_type': 'client_credentials',
        'client_id': input_id,
        'client_secret': input_secret
    }
    return data

def user_selection():
    while True:
        # Prompt user to select an action
        actions = ['top tracks', 'display song', 'playlist']
        text = input(f"Enter action {actions}: ").lower().strip()
        if text in actions:
            break
        else:
            print("Enter valid entry")
    return text

def get_artist_name():
    # Prompt user for artist name
    return input("Enter artist name ")

if __name__ == "__main__":
    main()
