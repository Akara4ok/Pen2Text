""" Class for page segmentation with a* algo """
import matplotlib.pyplot as plt
import cv2
from skimage.filters import sobel
from scipy.signal import savgol_filter
import numpy as np
from astar import astar


class AStarPageSegInference():
    """ Class for page segmentation with a* algo """
    def horizontal_projections(self, sobel_image: np.ndarray) -> np.ndarray:
        """ Calculate horizontal sum in image """
        return np.sum(sobel_image, axis=1)  

    def find_peak_regions(self,  hpp: np.ndarray, divider: int = 2) -> list:
        """ Returns list with peaks(possible line separator) """
        threshold = np.mean(hpp)
        peaks = []
        peaks_index = []
        for i, hppv in enumerate(hpp):
            if hppv < threshold:
                peaks.append([i, hppv])
        return peaks


    def get_hpp_walking_regions(self, peaks_index: list) -> list:
        """ Transform single peaks to windows """
        hpp_clusters = []
        cluster = []
        for index, value in enumerate(peaks_index):
            cluster.append(value)

            if index < len(peaks_index)-1 and peaks_index[index+1] - value > 1:
                hpp_clusters.append(cluster)
                cluster = []

            #get the last cluster
            if index == len(peaks_index)-1:
                hpp_clusters.append(cluster)
                cluster = []
                
        return hpp_clusters

    def path_exists(self, window_image: np.ndarray) -> bool:
        """ Check if there path between two points """
        #very basic check first then proceed to A* check
        if 0 in self.horizontal_projections(window_image):
            return True
        
        # padded_window = np.zeros((window_image.shape[0],1))
        # world_map = np.hstack((padded_window, np.hstack((window_image,padded_window)) ) )
        path = np.array(astar(window_image, (int(window_image.shape[0]/2), 0), (int(window_image.shape[0]/2), window_image.shape[1])))
        if len(path) > 0:
            return True
        
        return False

    def get_road_block_regions(self, nmap: np.ndarray) -> np.ndarray:
        """ get regions wich fully block astar """
        road_blocks = []
        needtobreak = False
        
        for col in range(nmap.shape[1]):
            start = col
            end = col+29
            if end > nmap.shape[1]-1:
                end = nmap.shape[1]-1
                needtobreak = True

            if self.path_exists(nmap[:, start:end]) == False:
                road_blocks.append(col)

            if needtobreak == True:
                break
                
        return road_blocks

    def group_the_road_blocks(self, road_blocks: list) -> list:
        """ Group single road blocker to groups """
        #group the road blocks
        road_blocks_cluster_groups = []
        road_blocks_cluster = []
        size = len(road_blocks)
        for index, value in enumerate(road_blocks):
            road_blocks_cluster.append(value)
            if index < size-1 and (road_blocks[index+1] - road_blocks[index]) > 1:
                road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
                road_blocks_cluster = []

            if index == size-1 and len(road_blocks_cluster) > 0:
                road_blocks_cluster_groups.append([road_blocks_cluster[0], road_blocks_cluster[len(road_blocks_cluster)-1]])
                road_blocks_cluster = []

        return road_blocks_cluster_groups

    def extract_line_from_image(self, image, lower_line, upper_line):
        """ Exctract text line between to boundary lines(upper and lower) """
        lower_boundary = np.min(lower_line[:, 0])
        upper_boundary = np.max(upper_line[:, 0])
        img_copy = np.copy(image)
        r, c = img_copy.shape
        for index in range(c-1):
            img_copy[0:lower_line[index, 0], index] = 0
            img_copy[upper_line[index, 0]:r, index] = 0
        
        return img_copy[lower_boundary:upper_boundary, :]


    def predict(self, x: np.ndarray) -> np.ndarray:
        """ Predict line img from page """
        pages = x
        result = []
    
        for img in pages:
            # img = img / 255
            # img = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 49, 35)
            sobel_image = sobel(img)
            hpp = self.horizontal_projections(sobel_image)

            hpp = savgol_filter(hpp, min(20, len(hpp)), min(7, len(hpp) - 1)) # window size 51, polynomial order 3

            peaks = self.find_peak_regions(hpp)

            peaks_index = np.array(peaks)[:,0].astype(int)

            segmented_img = np.copy(img)
            r,c = segmented_img.shape
            for ri in range(r):
                if ri in peaks_index:
                    segmented_img[ri, :] = 0

            hpp_clusters = self.get_hpp_walking_regions(peaks_index)
            if(len(hpp_clusters) < 3):
                result.append(img)
                continue

            binary_image = img

            for cluster_of_interest in hpp_clusters:
                nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
                road_blocks = self.get_road_block_regions(nmap)
                road_blocks_cluster_groups = self.group_the_road_blocks(road_blocks)
                #create the doorways
                for index, road_blocks in enumerate(road_blocks_cluster_groups):
                    window_image = nmap[:, road_blocks[0]: road_blocks[1]+10]
                    binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:][:][int(window_image.shape[0]/2),:] *= 0
                binary_image[-2:-1] = 0


            line_segments = []
            for i, cluster_of_interest in enumerate(hpp_clusters):
                nmap = binary_image[cluster_of_interest[0]:cluster_of_interest[len(cluster_of_interest)-1],:]
                path = np.array(astar(nmap, (int(nmap.shape[0]/2), 0), (int(nmap.shape[0]/2),nmap.shape[1]-1)))
                if len(path) > 0:
                    offset_from_top = cluster_of_interest[0]
                    path[:,0] += offset_from_top
                    line_segments.append(path)
            

            ## add an extra line to the line segments array which represents the last bottom row on the image
            last_bottom_row = np.flip(np.column_stack(((np.ones((img.shape[1],))*img.shape[0]), np.arange(img.shape[1]))).astype(int), axis=0)
            line_segments.append(last_bottom_row)

            line_images = []
            line_count = len(line_segments)
            for line_index in range(line_count-1):
                line_image = self.extract_line_from_image(img, line_segments[line_index], line_segments[line_index+1])
                line_images.append(line_image)
            
            result.extend(line_images[:-1])

        return result