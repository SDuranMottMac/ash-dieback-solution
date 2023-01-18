"""
This script identifies which videos have not been processed and give them back.
Therefore, they can be re-processed or further investigated
"""

# importing libraries
import numpy as np
import os


# Adding capabilities
class MissingVideos():
    """
    This class instantiates the object that identifies such videos
    """
    def __init__(self,input_dir,video_dir):
        # instantiating the object providing 2 arguments
        self.input_dir = input_dir
        self.video_dir = video_dir
    def unique(list1):
        x = np.array(list1)
        return(list(np.unique(x)))
    
    def listing_all_route_vids(self):
        """
        This method identifies all .svo files in the route folder
        """
        # instantiating an empty list
        all_vids = []
        # routing and obtaining .svo files
        for root, dirs, files in os.walk(self.video_dir, topdown=False):
            for item in files:
                if item.endswith(".svo"):
                    all_vids.append(item[:-4])
        # obtaining just unique files
        unique_vids = list(set(all_vids))
        return(unique_vids)
    
    def all_images(self):
        """
        This method identifies all .png files in the route folder
        """
        # instantiating an empty list
        all_imgs = []
        # routing and obtaining .svo files
        for root, dirs, files in os.walk(self.input_dir, topdown=False):
            for item in files:
                if item.endswith(".png"):
                    video_name = item.split("_detection_")[0]
                    if video_name not in all_imgs:
                        all_imgs.append(video_name)
        # obtaining just unique files
        unique_imgs = list(set((all_imgs)))
        return(unique_imgs)
    
    def all_shp_json(self):
        """
        This method identifies all the json files containing the .shp info
        """
        # instantiating an empty list
        all_json = []
        # routing and obtaining .svo files
        for root, dirs, files in os.walk(self.input_dir, topdown=False):
            for item in files:
                if item.endswith("_det.json"):
                    all_json.append(item[:-9])
        # obtaining just unique files
        unique_json = list(set((all_json)))
        return(unique_json)

    def non_processed_videos(self):
        """
        This method compares both and se
        """
        all_processed_vids = self.all_images()
        all_vids = self.listing_all_route_vids()
        all_json = self.all_shp_json()

        non_shp_videos = []
        for item in all_vids:
            if item not in all_json:
                non_shp_videos.append(item)
        return(non_shp_videos)
    
    def absolute_path_non_processed(self):
        non_shp_videos = self.non_processed_videos()
        rel_path_list = [] 
        for item in non_shp_videos:
            rel_path = os.path.relpath(item, start = self.video_dir)
            rel_path_list.append(rel_path)

        return(rel_path_list)
    
    def give_me_non_processes(self):
        rel_path = self.absolute_path_non_processed()
        with open(os.path.join(self.input_dir,'non_processes_videos.txt'), 'w') as f:
            for line in rel_path:
                f.write(f"{line}\n")

if __name__ == "__main__":

    project_checked = r""
    video_survey = r""

    # instantiating the object
    project1 = MissingVideos(project_checked,video_survey)
    # asking for the missing videos
    project1.give_me_non_processes()

