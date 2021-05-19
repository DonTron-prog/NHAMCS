#!/usr/bin/python3

import os
import pandas as pd

ed2011 = pd.read_csv(
    os.path.join('data', 'nhamcs2011.csv'), low_memory=False)
ed2012 = pd.read_csv(
    os.path.join('data', 'nhamcs2012.csv'), low_memory=False)
ed2013 = pd.read_csv(
    os.path.join('data', 'nhamcs2013.csv'), low_memory=False)
ed2014 = pd.read_csv(
    os.path.join('data', 'nhamcs2014.csv'), low_memory=False)
ed2015 = pd.read_csv(
    os.path.join('data', 'nhamcs2015.csv'), low_memory=False)
ed2016 = pd.read_csv(
    os.path.join('data', 'nhamcs2016.csv'), low_memory=False)
ed2017 = pd.read_csv(
    os.path.join('data', 'nhamcs2017.csv'), low_memory=False)
ed2018 = pd.read_csv(
    os.path.join('data', 'nhamcs2018.csv'), low_memory=False)

frames = [
    ed2011,
    ed2012,
    ed2013,
    ed2014,
    ed2015,
    ed2016,
    ed2017,
    ed2018,
]
dataset = pd.concat(frames, join='inner', ignore_index=True)

dataset.to_csv(os.path.join('data', 'nhamcs2011_2018.csv'))