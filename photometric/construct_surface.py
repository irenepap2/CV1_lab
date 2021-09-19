import numpy as np

def construct_surface(p, q, path_type='row'):

    '''
    CONSTRUCT_SURFACE construct the surface function represented as height_map
       p : measures value of df / dx
       q : measures value of df / dy
       path_type: type of path to construct height_map, either 'column',
       'row', or 'average'
       height_map: the reconstructed surface
    '''
    
    h, w = p.shape
    height_map = np.zeros([h, w])
    
    if path_type=='column':
        for i in range(1,h):
            height_map[i,0] = height_map[i-1,0] + q[i,0]

        for i in range(h):
            for j in range(1,w):
                height_map[i,j] = height_map[i,j-1] + p[i,j]
        """
        ================
        Your code here
        ================
        % top left corner of height_map is zero
        % for each pixel in the left column of height_map
        %   height_value = previous_height_value + corresponding_q_value
        
        % for each row
        %   for each element of the row except for leftmost
        %       height_value = previous_height_value + corresponding_p_value
        
        """
    elif path_type=='row':

        for i in range(1,w):
            height_map[0,i] = height_map[0,i-1] + p[0,i]

        for i in range(w):
            for j in range(1,h):
                height_map[j,i] = height_map[j-1,i] + q[j,i]
        
        """
        ================
        Your code here
        ================
        """
    elif path_type=='average':
        
        height_map_column = np.zeros([h, w])
        height_map_row = np.zeros([h, w])
        
        for i in range(1,h):
            height_map_column[i,0] = height_map_column[i-1,0] + q[i,0]

        for i in range(h):
            for j in range(1,w):
                height_map_column[i,j] = height_map_column[i,j-1] + p[i,j]

        for i in range(1,w):
            height_map_row[0,i] = height_map_row[0,i-1] + p[0,i]

        for i in range(w):
            for j in range(1,h):
                height_map_row[j,i] = height_map_row[j-1,i] + q[j,i]
        
        height_map = (height_map_column + height_map_row)/2
        """
        ================
        Your code here
        ================
        """
        
    return height_map
        
