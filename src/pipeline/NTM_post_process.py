from osgeo import ogr, osr
from tqdm import tqdm

# os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "3000"
def auto_processing(first_time=True, detection_shp=None,ntm_shp=None,road_shp=None):
    # reading the detection and NTM shapefiles
    detections = ogr.Open(detection_shp, 1)
    NTM = ogr.Open(ntm_shp, 1)
    # obtaining the corresponding layer
    det_layer = detections.GetLayer()
    ntm_layer = NTM.GetLayer()

    src = det_layer.GetSpatialRef()
    tgt = ntm_layer.GetSpatialRef()
    # whilst great to have them in BNG (27700), it's more important for them to be the same for intersections.
    tgt = osr.SpatialReference()
    tgt.ImportFromEPSG(27700)
    transform = osr.CoordinateTransformation(src, tgt)

    print("Creating Health field")
    if first_time:
        newfield = ogr.FieldDefn("Health", ogr.OFTString)
        det_layer.CreateField(newfield)

    print("Turning Dieback into Crown Health")
    dieback_rel = {0:r"100%-75% live crown", 1:r"75-50% live crown", 2:r"50-25% live crown", 3:r"25-0% live crown"}
    for feature in det_layer:
        dieback_lvl = feature.GetField("Dieback")
        feature.SetField("Health", dieback_rel[dieback_lvl])
        det_layer.SetFeature(feature)
        # det_layer.DeleteField(1)
    det_layer.ResetReading()

    print("Creating NTM field")
    if first_time:
        newfield = ogr.FieldDefn("NTM_ID", ogr.OFTString)
        newfield_2 = ogr.FieldDefn("NTM_H", ogr.OFTInteger)
        newfield_3 = ogr.FieldDefn("Max_H", ogr.OFTInteger)
        newfield_4 = ogr.FieldDefn("Height_Bnd", ogr.OFTString)
        det_layer.CreateField(newfield)
        det_layer.CreateField(newfield_2)
        det_layer.CreateField(newfield_3)
        det_layer.CreateField(newfield_4)

    # intersections = {} # detection FID: NTM FID
    for feat1 in tqdm(det_layer, "Finding Intersections between detections and NTM"):
        geom1 = feat1.GetGeometryRef()
        # transform from one coord system to the other, might not be needed but worth catching if it is
        # geom1.Transform(transform)
        count = 0
        intersection = False
        while intersection == False:
            feat2 = ntm_layer.GetNextFeature()
            if count == len(ntm_layer):
                break
            elif geom1.Intersects(feat2.GetGeometryRef()):
                intersection = True
                # intersections[feat1.GetFID()] = feat2.GetFID()

                feat1.SetField("NTM_ID", str(feat2.GetField("NTM_ID")))
                feat1.SetField("NTM_H", int(feat2.GetField("MAX")))

                if float(feat1.GetField("MM_Height")) < float(feat2.GetField("MAX")):
                    feat1.SetField("Max_H", int(feat2.GetField("MAX")))
                else:
                    feat1.SetField("Max_H", int(feat1.GetField("MM_Height")))

            else:
                feat1.SetField("NTM_ID", None)
                feat1.SetField("NTM_H", None)
                feat1.SetField("Max_H", int(feat1.GetField("MM_Height")))

            height = int(feat1.GetField("Max_H"))

            if height >= 0 and height < 5:
                feat1.SetField("Height_Bnd", "0-5 m")
            elif height >= 5 and height < 10:
                feat1.SetField("Height_Bnd", "5-10 m")
            elif height >= 10 and height < 15:
                feat1.SetField("Height_Bnd", "10-15 m")
            elif height >= 15 and height < 20:
                feat1.SetField("Height_Bnd", "15-20 m")
            else:
                feat1.SetField("Height_Bnd", "> 20m ")
        
            count+=1

        det_layer.SetFeature(feat1)
        ntm_layer.ResetReading()

    print("Opening Highways shapefile")
    highways = ogr.Open(road_shp)
    ways_layer = highways.GetLayer()
    # newGeometry = ways_layer[0].GetGeometryRef()

    # newGeometry = None
    # for feature in tqdm(ways_layer, desc="Merging indivudal road polygons"):
    #     geometry = feature.GetGeometryRef()
    #     if newGeometry is None:
    #         newGeometry = geometry.Clone()
    #     else:
    #         if geometry != None:
    #             newGeometry = newGeometry.Union(geometry)

    # src = ways_layer.GetSpatialRef()
    # tgt = ntm_layer.GetSpatialRef()
    # transform = osr.CoordinateTransformation(src, tgt)

    if first_time:
        fall_area = ogr.FieldDefn("Fall_Area", ogr.OFTReal)
        det_layer.CreateField(fall_area)

    det_layer.ResetReading()
    for feat1 in tqdm(det_layer, "Turning Height into Fall Radius"):
        center_geom = feat1.GetGeometryRef()
        # center_geom.Transform(transform)
        radius = feat1.GetField("Max_H")
        if radius != None:
            bufferDefn = center_geom.Buffer(float(radius), 50)
            bufferDefnPolygon = bufferDefn.GetGeometryRef(0)

            actualCircle = ogr.ForceToPolygon(bufferDefnPolygon)

            # create layer of circles and then intersect the highways and circles layers.
            # iterate through the intersection and pull each one.
            ways_layer.ResetReading()
            for feat2 in ways_layer:
                geometry = feat2.GetGeometryRef()
                if geometry != None:
                    if actualCircle.Intersects(geometry):
                        fallArea = actualCircle.Intersection(geometry)
                        if fallArea != None:
                            feat1.SetField("Fall_Area", fallArea.GetArea())
                    det_layer.SetFeature(feat1)

    # go through our detections layer, take the point and height/radius to create a
    # circle and then do intersection with the highways layer. REMEMBER SPATIAL REF!! and ResetReading()


if __name__ == "__main__":
    auto_processing(first_time=True, detection_shp=r"C:\Users\mun96437\Documents\NTM_Test\glasgow_shp\ADDR1P2.shp",
                     ntm_shp=r"C:\Users\mun96437\Documents\NTM_Test\merge_32630\glasgow\glasgow.shp",
                     road_shp=r"C:\Users\mun96437\Documents\NTM_Test\Adopted_Highways_32630\Glasgow\merged\OS_Highways_and_Footways_32630_merge.shp")