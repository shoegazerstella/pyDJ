#!/usr/bin/env python

import os
import librosa
import numpy as np
from glob import glob
from pydub import AudioSegment
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import argparse


config = {'SR': 22050,
        'OFFSET': 5,
        'DURATION': 20,
        'CROSSFADE': 10}


def load_audio(audioPath):
    y, sr = librosa.load(audioPath, duration=config['DURATION'], sr=config['SR'])
    
    audiosamples = config['SR'] * config['DURATION']

    if y.shape[0] < audiosamples:
        y = np.pad(y, (0, audiosamples), 'constant')
        
    y = y[:audiosamples]
          
    return y, sr

def get_tempo(y, sr):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    return tempo, beats

def get_mfcc(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return mfccs

def compute_features(inputDir, audiofiles):

    features = []
    tempos = []
    songs_list = []

    for audio in tqdm(audiofiles):
        
        try:
            y, sr = load_audio(os.path.join(inputDir, audio))
            tempo, _ = get_tempo(y, sr)
            mfccs = get_mfcc(y,sr)

            tempos.append(tempo)
            mfccs = np.ravel(mfccs)
            features.append(mfccs)
            songs_list.append(audio)
            
        except Exception as e:
            print(e)
        
    features = np.array(features)
    
    return features, songs_list, tempos

def computeNN(features):
    nbrs = NearestNeighbors(n_neighbors=features.shape[0], algorithm='ball_tree').fit(features)
    return nbrs

def get_min_bpm_index(tempos):
    min_bpm_index = tempos.index(min(tempos))
    return min_bpm_index

def create_playlist(results, outputDir):

    playlist_songs = [AudioSegment.from_file(audio_file) for audio_file in results]

    first_song = playlist_songs.pop(0)

    beginning_of_song = first_song.fade_in(2000)

    playlist = beginning_of_song

    CROSSFADE = config['CROSSFADE']
    for song in tqdm(playlist_songs):
        
        if len(song) <= config['CROSSFADE'] * 1000:
            CROSSFADE = 1

        # We don't want an abrupt stop at the end, so let's do a 10 second crossfades
        playlist = playlist.append(song, crossfade=(CROSSFADE * 1000))

    # fade out the end of the last song
    playlist = playlist.fade_out(30)

    # mixtape lenght ( len(audio_segment) returns milliseconds )
    playlist_length = len(playlist) / (1000*60)
    print('playlist duration:', playlist_length)

    # save
    out_f = open(os.path.join(outputDir, 'playlist.mp3'), 'wb')
    playlist.export(out_f, format='wav')
    return

def main(inputDir, outputDir, neighs):

    audiofiles = os.listdir(inputDir)

    if neighs is None:
        neighs = 10

    if neighs < len(audiofiles):
        neighs = len(audiofiles)

    print('Computing features..')
    features, songs_list, tempos = compute_features(inputDir, audiofiles)

    print('Computing NNs')
    nbrs = computeNN(features)

    # get song with lowest tempo
    min_bpm_index = get_min_bpm_index(tempos)

    # start 
    distances, indices = nbrs.kneighbors(features[min_bpm_index].reshape(1, -1))

    results = []
    for i in indices[0]:
        results.append(songs_list[i])

    results = results[:neighs]
    
    print('PLAYLIST')
    for i in results:
        print(i)

    results = [os.path.join(inputDir, i) for i in results]

    print('Creating playlist..')
    create_playlist(results, outputDir)

    return

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='pyDJ - automatic mixtapes generation')
    parser.add_argument('-i', '--inputDir', type=str, required=True)
    parser.add_argument('-o', '--outputDir', type=str, required=True)
    parser.add_argument('-n', '--neighs', default=None, type=int, required=False)
    args = parser.parse_args()

    indir = args.inputDir
    outdir = args.outputDir
    n = args.neighs

    main(indir, outdir, n)