from pyproj import Proj
import pandas as pd
###configuration####
acceptedFilename = "/data6/tziolkowski/data/raw_txt/running_example/MSM88_accepted_example.txt"
rejectedFilename = "/data6/tziolkowski/data/raw_txt/running_example/MSM88_rejected_example.txt"
exportCsvName="/data6/tziolkowski/data/raw_txt/running_example/combined_update.csv"
exportFeatherName="/data6/tziolkowski/data/raw_txt/running_example/combinedFeather"



def recalculateCoordinates(df):
    print("data frame created")
    # project use and values partly from here https://ocefpaf.github.io/python4oceanographers/blog/2013/12/16/utm/)
    myProj = Proj(proj='utm', zone=26, ellps='WGS84', preserve_units=False)
    print("project created")
    df['lon'], df['lat'] = myProj(df['east'].values, df['north'].values, inverse=True)
    print("coordinates calculated")
    result = df.loc[:, ['lat', 'lon', 'depth']]
    return result


accepted = recalculateCoordinates(pd.read_csv(acceptedFilename, delimiter=';', names=['east', 'north', 'depth']))
# create outlier attribute 0=accepted 1=rejected
accepted['outlier'] = 0
rejected = recalculateCoordinates(pd.read_csv(rejectedFilename, delimiter=';', names=['east', 'north', 'depth']))
rejected['outlier'] = 1
# concatenate both files.
result = pd.concat([accepted, rejected])
result = result.sort_values(by=['lon'])
result.reset_index(inplace=True)
result['index'] = result.index
result.to_feather(exportFeatherName)
result.to_csv(exportCsvName, index=False)
print('finished')
