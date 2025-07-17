'''

Author : yunanhou

Date : 2023/11/26

Function : This script is used to parse the longitude, latitude and altitude position information of the satellite at
           a specified time from TLE data.

'''
from skyfield.api import load, EarthSatellite

# analyze the satellite's latitude, longitude and altitude position based on TLE data and specified time
# Parameter :
# TLE : The TLE data of the constellation can be downloaded from the website
#       https://celestrak.org/NORAD/elements/index.php?FORMAT=2le. The parameter type is list, and each element in the
#       list is a row of TLE data, and the TLE data format is 2LE
# year, month, day, hour, minute, second : to calculate the time corresponding to the satellite position
def get_satellite_position(TLE, year, month, day, hour, minute, second):
    # load ephemeris data
    #planets = load('de421.bsp')
    ts = load.timescale()
    # specify a specific UTC time point
    t = ts.utc(year, month, day, hour, minute, second)
    # parse TLE data and obtain satellite position information
    num_satellites = len(TLE) // 2
    satellite_positions = []
    for i in range(0, len(TLE), 2):
        tle_line1 = TLE[i].strip()
        tle_line2 = TLE[i + 1].strip()
        # create satellite object
        satellite = EarthSatellite(tle_line1, tle_line2, 'SAT', ts)
        # calculate satellite position
        topocentric = satellite.at(t)
        subpoint = topocentric.subpoint()
        # get the geographical coordinates of a satellite (longitude, latitude, altitude)
        longitude = subpoint.longitude.degrees
        latitude = subpoint.latitude.degrees
        altitude = subpoint.elevation.km
        satellite_positions.append((longitude, latitude, altitude))
    return satellite_positions

if __name__ == "__main__":
    sample_tle = [
        "1 25544U 98067A   23329.37916667  .00002182  00000-0  40858-4 0  9990",
        "2 25544  51.6461 339.7939 0001258  92.2360 267.9496 15.49309620426818"
    ]

    positions = get_satellite_position(sample_tle, 2023, 11, 26, 12, 0, 0)
    print(f"卫星位置: {positions}")