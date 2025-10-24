import pvlib
import pandas as pd
from pvlib.location import Location

# Define site (Cape Town example)
site = Location(-33.9249, 18.4241, 'Africa/Johannesburg', 25, 'Cape Town')

# Make a time range (one day, hourly)
times = pd.date_range('2025-01-01', '2025-01-02', freq='1h', tz=site.tz)

# Get solar position for those times
solpos = site.get_solarposition(times)

print(solpos.head())

