import numpy as np
def rect_to_bb(rect):
    """
    input: bounding box predicted by dlib and convert
    to the format (x,y,w,h) 
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

# change to np array
def shape_to_np(shape, dtype = "int"):
    #initialize list of (x,y) coordinates
    co = np.zeros((68, 2), dtype=dtype)
    #loop over 68 facial landmarks and convert
    # them to numpy array
    for i in range(0,68):
        co[i] = (shape.part(i).x, shape.part(i).y)
    return co