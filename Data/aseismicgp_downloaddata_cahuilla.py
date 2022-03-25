from obspy.clients.fdsn import Client
from obspy.core import UTCDateTime
import numpy as np

client = Client("SCEDC")

#Caihulla swarm larger magnitude events

starttime = UTCDateTime("2016-01-01")
endtime = UTCDateTime("2019-12-31")
minlatitude = 33.42043
maxlatitude = 33.58032
minlongitude = -116.84723
maxlongitude = -116.68990
minmagnitude = 1.71 #estimate for SCSN magnitude of completeness


cat = client.get_events(starttime=starttime, 
                        endtime=endtime,
                        minlatitude=minlatitude,
                        maxlatitude=maxlatitude,
                        minlongitude=minlongitude,
                        maxlongitude=maxlongitude,
                        minmagnitude=minmagnitude)

caihulla_data = np.array([[ev.origins[0].time.matplotlib_date for ev in cat],
                          [ev.magnitudes[0].mag for ev in cat]])

np.savetxt("cahuilla_data.txt", caihulla_data.T)  

#Pre-swarm data

starttime = UTCDateTime("2000-01-01")
endtime = UTCDateTime("2016-01-01")
minlatitude = 33.42043
maxlatitude = 33.58032
minlongitude = -116.84723
maxlongitude = -116.68990
minmagnitude = 1.71 #estimate for SCSN magnitude of completeness

cat = client.get_events(starttime=starttime, 
                        endtime=endtime,
                        minlatitude=minlatitude,
                        maxlatitude=maxlatitude,
                        minlongitude=minlongitude,
                        maxlongitude=maxlongitude,
                        minmagnitude=minmagnitude)

caihulla_preswarm_data = np.array([[ev.origins[0].time.matplotlib_date for ev in cat],
                          [ev.magnitudes[0].mag for ev in cat]])

np.savetxt("cahuilla_preswarm_data.txt", caihulla_preswarm_data.T) 