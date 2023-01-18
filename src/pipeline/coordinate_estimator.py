import os



def inverse_distance(distance_list,latitude_list,longitude_list):
    '''
    This method defines the averaged coordinates using the inverse of the distance for each pair of coordinate
    '''
    # getting all the distances in absolute value
    distances = [abs(ele) for ele in distance_list]
    # getting the sum of the distances
    sum_dist = sum(distances)
    # getting the fraction that each distance represent of the total
    fraction_distances = [ele/sum_dist for ele in distances]
    # getting the inverse of the fraction
    inv_distances = [1/ele for ele in fraction_distances]
    # calculating the sum of all inverse fractions
    sum_inv = sum(inv_distances)
    # assigning weights accordingly
    weights = [ele/sum_inv for ele in inv_distances]

    final_latitude = 0
    final_longitude = 0
    for i in range(len(distance_list)):

        final_latitude = final_latitude + weights[i]*latitude_list[i]
        final_longitude = final_longitude + weights[i]*longitude_list[i]
        
    return(final_latitude,final_longitude)


def last_estimations(latitude_list,longitude_list,k):
    '''
    This method defines the averaged coordinates using the k closest estimations
    '''
    # we need to get the last k items
    latitudes = latitude_list[-k:]
    longitudes = longitude_list[-k:]

    # getting the average
    avg_latitude = sum(latitudes) / len(latitudes)
    avg_longitude = sum(longitudes) / len(longitudes)

    # output
    return(avg_latitude,avg_longitude)
def last_estimations_no_outliers(distance,latitude_list,longitude_list,k=5,avg_dist=20):
    '''
    This method defines the averaged coordinates using the k closest estimations
    '''
    # First, we filter potential outliers or errors
    distances = []
    latitudes_lists = []
    longitudes_lists = []
    # if the distance used for calculating the coordinates is larger than the distance theshold used - we do not considered them
    for ixx,ele in enumerate(distance):
        if abs(int(ele)) < abs(int(avg_dist)):
            # the lists of distance and coordinates are populated
            distances.append(abs(ele))
            latitudes_lists.append(latitude_list[ixx])
            longitudes_lists.append(longitude_list[ixx])

    # Then, we use the resulting list to estimate the latitude longitude
    if len(distances) >= (k+2): 
        # we need to get the last k+2 items
        latitudes = latitudes_lists[-k-2:]
        longitudes = longitudes_lists[-k-2:]
    elif len(distances) == (k+1): 
        # we need to get the last k+1 items
        latitudes = latitudes_lists[-k-1:]
        longitudes = longitudes_lists[-k-1:]
    elif len(distances) == (k+0):
        latitudes = latitude_list[-k:]
        longitudes = longitude_list[-k:]
    else:
        latitudes = latitudes_lists[-len(distances):]
        longitudes = longitudes_lists[-len(distances):]

    # getting the average
    avg_latitude = sum(latitudes) / len(latitudes)
    avg_longitude = sum(longitudes) / len(longitudes)

    # output
    return(avg_latitude,avg_longitude)

def last_rear_estimations_no_outliers(distance,latitude_list,longitude_list,k=5,avg_dist=20):
    '''
    This method defines the averaged coordinates using the k closest estimations
    '''
    # First, we filter potential outliers or errors
    distances = []
    latitudes_lists = []
    longitudes_lists = []
    # if the distance used for calculating the coordinates is larger than the distance theshold used - we do not considered them
    for ixx,ele in enumerate(distance):
        if abs(int(ele)) < abs(int(avg_dist)):
            # the lists of distance and coordinates are populated
            distances.append(abs(ele))
            latitudes_lists.append(latitude_list[ixx])
            longitudes_lists.append(longitude_list[ixx])

    # Then, we use the resulting list to estimate the latitude longitude
    if len(distances) >= (k+2): 
        # we need to get the last k+2 items
        latitudes = latitudes_lists[:k+2]
        longitudes = longitudes_lists[:k+2]
    elif len(distances) == (k+1): 
        # we need to get the last k+1 items
        latitudes = latitudes_lists[:k+1]
        longitudes = longitudes_lists[:k+1]
    elif len(distances) == (k+0):
        latitudes = latitude_list[:k]
        longitudes = longitude_list[:k]
    else:
        latitudes = latitudes_lists[:len(distances)]
        longitudes = longitudes_lists[:len(distances)]

    # getting the average

    avg_latitude = sum(latitudes) / len(latitudes)
    avg_longitude = sum(longitudes) / len(longitudes)
    

    # output
    return(avg_latitude,avg_longitude)

if __name__ == "__main__":
    distance = [20, 16, 15]
    latitude_list = [53.32324324,53.32324335,53.32324329]
    longitude_list = [-3.532456,-3.532459,-3.532464]

    lat,long = last_estimations(latitude_list,longitude_list,k=2)
    #lat,long = inverse_distance(distance_list=distance,latitude_list=latitude_list,longitude_list=longitude_list)
    print(lat,long)