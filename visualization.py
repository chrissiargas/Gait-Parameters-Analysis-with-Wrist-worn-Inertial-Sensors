from plotnine.ggplot import *
from plotnine.geoms import *
from plotnine.scales import *
from plotnine.labels import *
import pandas as pd
from configParser import Parser
import os

SUBJECT_ID = '20'

ACTIVITY = 'outdoor_walknrun'

start = 1
end = 500

config = Parser()
config.get_args()

if config.dataset == 'MAREA':
    path = os.path.join(
        os.path.expanduser('~'),
        config.path,
        'MAREA',
    )

# Read in the data
DATA_PATH = os.path.join(path, 'Data_csv format')

df = pd.read_csv(os.path.join(DATA_PATH, 'Sub_' + SUBJECT_ID + '.csv'), header=0)

k1 = df[df[ACTIVITY] == 1]
k1.reset_index(level=0, inplace=True)

k1['LF_HS_mult'] = k1.LF_HS*30
k1['RF_HS_mult'] = k1.RF_HS*30
k1['LF_TO_mult'] = k1.LF_TO*(-30)
k1['RF_TO_mult'] = k1.RF_TO*(-30)

place = 'LF'
plot = ggplot(mapping=aes(x='index'), data=k1[start:end]) +\
    geom_line(aes(y='accX_' + place), color='blue') +\
    geom_line(aes(y='accY_' + place), color='red') +\
    geom_line(aes(y='accZ_' + place), color='green')

print(plot)

plot = ggplot(mapping=aes(x='index'), data=k1[start:end]) +\
    geom_line(aes(y='accX_' + place), color='blue') +\
    geom_line(aes(y='accY_' + place), color='red') +\
    geom_line(aes(y='accZ_' + place), color='green') +\
    geom_point(aes(y='LF_HS_mult'), color='steelblue', size=3) +\
    geom_point(aes(y='LF_TO_mult'), color='blue', size=3) +\
    geom_point(aes(y='RF_HS_mult'), color='hotpink', size=3) +\
    geom_point(aes(y='RF_TO_mult'), color='pink', size=3) +\
    scale_y_continuous(limits=(-40, 40)) +\
    ggtitle('Left Foot Acceleration') +\
    xlab('index') +\
    ylab('Acceleration')

print(plot)

