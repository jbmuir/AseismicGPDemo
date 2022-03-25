from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import numpy as np

client = Client("SCEDC")

#Caihulla swarm larger magnitude events

starttime = UTCDateTime("2019-07-01")
endtime = UTCDateTime("2019-07-15")
minlatitude = 35.4699
maxlatitude = 36.0361
minlongitude = -117.9435
maxlongitude = -117.2132
minmagnitude = 3.0 #anything smaller and the catalog will be too large


cat1 = client.get_events(starttime=starttime, 
                        endtime=endtime,
                        minlatitude=minlatitude,
                        maxlatitude=maxlatitude,
                        minlongitude=minlongitude,
                        maxlongitude=maxlongitude,
                        minmagnitude=minmagnitude)

ridgecrest_data = np.array([[ev.origins[0].time.matplotlib_date for ev in cat1],
                          [ev.magnitudes[0].mag for ev in cat1]])

np.savetxt("ridgecrest_data.txt", ridgecrest_data.T)  

#Pre-swarm data

starttime = UTCDateTime("2000-01-01")
endtime = UTCDateTime("2019-07-01")

cat2 = client.get_events(starttime=starttime, 
                        endtime=endtime,
                        minlatitude=minlatitude,
                        maxlatitude=maxlatitude,
                        minlongitude=minlongitude,
                        maxlongitude=maxlongitude,
                        minmagnitude=minmagnitude)

ridgecrest_preswarm_data = np.array([[ev.origins[0].time.matplotlib_date for ev in cat2],
                          [ev.magnitudes[0].mag for ev in cat2]])

np.savetxt("ridgecrest_preswarm_data.txt", ridgecrest_preswarm_data.T) 


poissonian_24h_estimate = len(cat2)/((endtime-starttime)/(3600*24))