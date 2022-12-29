""" Class for line segmentation with statistical methods """
import numpy as np

class StatLineSegInference():
    """ Class for line segmentation with statistical methods """
    def find_mean_whitespace_length(self, vertical_projection: np.ndarray) -> float:
        ## we will go through the vertical projections and 
        ## find the sequence of consecutive white spaces in the image
        whitespace_lengths = []
        whitespace = 0
        for vp in vertical_projection:
            if vp == 0:
                whitespace = whitespace + 1
            elif vp != 0:
                if whitespace != 0:
                    whitespace_lengths.append(whitespace)
                whitespace = 0 # reset whitepsace counter. 
        avg_white_space_length = np.mean(whitespace_lengths)
        return avg_white_space_length

    def divide_line_to_words(self, vertical_projection: np.ndarray, avg_white_space_length: float) -> np.ndarray:
        whitespace_length = 0
        divider_indexes = []
        for index, vp in enumerate(vertical_projection):
            if vp == 0:
                whitespace_length = whitespace_length + 1
            elif vp != 0:
                if whitespace_length != 0 and whitespace_length > avg_white_space_length:
                    divider_indexes.append(index-int(whitespace_length/2))
                    whitespace_length = 0 # reset it
        return divider_indexes

    def predict(self, x: np.ndarray) -> np.ndarray:
        lines = x
        result = []
        for line_img in lines:
            vertical_projection = np.sum(line_img, axis=0)

            avg_mean_whitespace = self.find_mean_whitespace_length(vertical_projection)
            divider_indexes = self.divide_line_to_words(vertical_projection, avg_mean_whitespace)

            divider_indexes = np.array(divider_indexes)
            dividers = np.column_stack((divider_indexes[:-1],divider_indexes[1:]))

            for window in dividers:
                result.append(line_img[:,window[0]:window[1]])
        
        return result