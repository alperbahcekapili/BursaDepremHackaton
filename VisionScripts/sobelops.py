"""
@file sobel_demo.py
@brief Sample code using Sobel and/or Scharr OpenCV functions to make a simple Edge Detector
"""
import sys
import cv2 as cv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from optimal_road_finder import RoadFinder

def zoom_factory(ax,base_scale = 2.):
    def zoom_fun(event):
        # get the current x and y limits
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()
        cur_xrange = (cur_xlim[1] - cur_xlim[0])*.5
        cur_yrange = (cur_ylim[1] - cur_ylim[0])*.5
        xdata = event.xdata # get event x location
        ydata = event.ydata # get event y location
        if event.button == 'up':
            # deal with zoom in
            scale_factor = 1/base_scale
        elif event.button == 'down':
            # deal with zoom out
            scale_factor = base_scale
        else:
            # deal with something that should never happen
            scale_factor = 1
            print (event.button)
        # set new limits
        ax.set_xlim([xdata - cur_xrange*scale_factor,
                     xdata + cur_xrange*scale_factor])
        ax.set_ylim([ydata - cur_yrange*scale_factor,
                     ydata + cur_yrange*scale_factor])
        plt.draw() # force re-draw

    fig = ax.get_figure() # get the figure of interest
    # attach the call back
    fig.canvas.mpl_connect('scroll_event',zoom_fun)

    #return the function
    return zoom_fun

def main(argv):
    
    window_name = ('Sobel Demo - Simple Edge Detector')
    scale = 2
    delta = 0
    ddepth = cv.CV_16S
    
    
    if len(argv) < 1:
        print ('Not enough parameters')
        print ('Usage:\nmorph_lines_detection.py < path_to_image >')
        return -1
    # Load the image
    src = cv.imread(argv[0], cv.IMREAD_COLOR)
    # Check if image is loaded fine
    if src is None:
        print ('Error opening image: ' + argv[0])
        return -1
    cv.imshow("Source", src)
    
    # show detections 
    dets = cv.imread("Detections.png")
    cv.imshow("Yolov8 Detections",dets)

    src2 = cv.GaussianBlur(src, (3, 3), 0)
    
    
    gray = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)
    
 
    grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    # # Gradient-Y
    #grad_y = cv.Scharr(gray,ddepth,0,1)
    grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
    
    
    abs_grad_x = cv.convertScaleAbs(grad_x)
    abs_grad_y = cv.convertScaleAbs(grad_y)
    
    
    sobeled = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
    et,thresh1 = cv.threshold(sobeled,160,255,cv.THRESH_BINARY)
    thresh1_v1 = thresh1.copy()
    cv.imshow("After Tresholding Operation", thresh1_v1)
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE,(40,40))
    res = cv.morphologyEx(thresh1_v1,cv.MORPH_CLOSE,kernel)
    cv.imshow('After Closing Operation',res)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(55,55))
    openres = cv.morphologyEx(res,cv.MORPH_OPEN,kernel)
    openres = 255-openres



    # burada optimal yol bulma kodu
    cv.imshow('After Opening Operation',openres)

    r = RoadFinder(openres)

    r.get_grid_values()
    r.filter_nodes()
    window_name = 'Start and target pooints for path extraction'
    
    
    
    # Radius of circle
    radius = 10
    
    # Blue color in BGR
    color = (255, 0, 0)
    
    # Line thickness of 2 px
    thickness = 10
    
    image = src.copy()
    # visualize filtered nodes
    
    
    

    start, target, route = r.generate_graph()

    for n in r.filtered_nodes:
        (x,y) = (n.locx*r.grid_h, n.locy*r.grid_w)
        # Center coordinates
        center_coordinates = (int(r.grid_h/2)+y, int(r.grid_w/2) + x)
        print(center_coordinates)
        image = cv.circle(image, center_coordinates, radius, (0,255,0), thickness)


    for n in route:
        (x,y) = (n.locx*r.grid_h, n.locy*r.grid_w)
        # Center coordinates
        center_coordinates = (int(r.grid_h/2)+y, int(r.grid_w/2) + x)
        print(center_coordinates)
        image = cv.circle(image, center_coordinates, radius, color, thickness)
    
    (x,y) = (start.locx*r.grid_h, start.locy*r.grid_w)
    # Center coordinates
    center_coordinates = (int(r.grid_h/2)+y, int(r.grid_w/2) + x)
    print(center_coordinates)
    image = cv.circle(image, center_coordinates, radius, (0,0,255), thickness)
    
    (x,y) = (target.locx*r.grid_h, target.locy*r.grid_w)
    # Center coordinates
    center_coordinates = (int(r.grid_h/2)+y, int(r.grid_w/2) + x)
    print(center_coordinates)
    image = cv.circle(image, center_coordinates, radius, (0,0,255), thickness)
    
    
    for n in r.filtered_nodes:
        (x,y) = (n.locx*r.grid_h, n.locy*r.grid_w)
        # Center coordinates
        center_coordinates = (int(r.grid_h/2)+y, int(r.grid_w/2) + x)
        print(center_coordinates)
        image = cv.circle(image, center_coordinates, radius, (0,255,0), thickness)

    
    
# Displaying the image 
    cv.imshow("Optimal Route", image) 
    output = cv.connectedComponentsWithStats(openres, 8, cv.CV_32S)
    (numLabels, labels, stats, centroids) = output
    for i in range(0, numLabels):
        # if this is the first component then we examine the
        # *background* (typically we would just ignore this
        # component in our loop)
        if i == 0:
            text = "examining component {}/{} (background)".format(
                i + 1, numLabels)
        # otherwise, we are examining an actual connected component
        else:
            text = "examining component {}/{}".format( i + 1, numLabels)
        # print a status message update for the current connected
        # component
        print("[INFO] {}".format(text))
        # extract the connected component statistics and centroid for
        # the current label
        x = stats[i, cv.CC_STAT_LEFT]
        y = stats[i, cv.CC_STAT_TOP]
        w = stats[i, cv.CC_STAT_WIDTH]
        h = stats[i, cv.CC_STAT_HEIGHT]
        area = stats[i, cv.CC_STAT_AREA]
        (cX, cY) = centroids[i]

        output = src.copy()
        cv.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)
        componentMask = (labels == i).astype("uint8") * 255
        # show our output image and connected component mask
        cv.imshow("Connected Component on Source", output)
        cv.imshow("Connected Component", componentMask)
        cv.waitKey(0)
    
    contours, hierarchy = cv.findContours(openres, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    mylist=[]
    polygons=[]
    for poly in contours:
        poly=poly.tolist()
        for point in poly:
            tuples = [tuple(x) for x in point]
            
            mylist.extend(tuples)
        polygons.append(plt.Polygon(mylist,ec="k"))
        mylist=[]        
            

    print(mylist)
    #tuples = [tuple(x) for x in contours]
    #print(contours)

    fig,ax=plt.subplots()
    ax.plot(range(10))
    scale = 1.5
    f = zoom_factory(ax,base_scale = scale)
    # bura
    with open('loclist.pkl','rb') as f:
        whole_list=pkl.load(f)
    for poly in polygons:
        ax.add_patch(poly)
    for element in whole_list:
        carcar = plt.Polygon(element,ec='k')
        carcar.set_color([0,0.5,0])
        ax.add_patch(carcar)    

    for n in route:
        (x,y) = (n.locx*r.grid_h, n.locy*r.grid_w)
        center_coordinates = (int(r.grid_h/2)+y, int(r.grid_w/2) + x)
        patch = matplotlib.patches.Circle(center_coordinates, radius=10, transform=ax.transData)
        patch.set_color([0.5,0,0])
        ax.add_patch(patch)    

    ax.set_ylim(0,1000)
    ax.set_xlim(0,1000)


    
    ax.invert_yaxis()

    # kontur = cv.drawContours(src, contours, -1, (0,255,0), 10)
    # print(kontur)
    
    # subtracted = cv.subtract(src,kontur)
    # cv.imshow('contour',subtracted)

    #kernel = np.ones((15, 15), np.uint8)
    
    # The first parameter is the original image,
    # kernel is the matrix with which image is
    # convolved and third parameter is the number
    # of iterations, which will determine how much
    # you want to erode/dilate a given image.

    #img_dilation = cv.dilate(thresh1_v1, kernel, iterations=1)
    #img_erosion = cv.erode(res, kernel, iterations=1)
    #cv.imshow('dilation',img_dilation)
    #cv.imshow('erosion',img_erosion)


    # gridler al 
    (h,w, _) = src.shape
    print("Height of the image: ", h)
    print("Width of the image: ", w)
    grid_h, grid_w = min(int(h/10)+1,200), min(int(w/10)+1, 200)
    for i in range(0, h, grid_h):
        for j in range(0,w, grid_w):
            grid_copy = thresh1[i:i+grid_h,j:j+grid_w].copy()
            grid_avg = np.mean(grid_copy)
            
            # now split each grid into 9 pieces
            inner_partitions = 3
            child_grid_h = int((grid_h+2) / inner_partitions)
            child_grid_w = int((grid_w+2) / inner_partitions)


            
            child_avgs = []
            for h_temp in range(inner_partitions):
                inner_temp = []
                for w_temp in range(inner_partitions):
                    tmp = grid_copy[h_temp * child_grid_h :(h_temp+1) * child_grid_h, 
                                    w_temp * child_grid_w :(w_temp+1) * child_grid_w]
                    
                    tmp_avg = np.mean(tmp) * grid_avg/255
                    #tmp_avg = 0 if tmp_avg < 60 else tmp_avg
                    inner_temp.append(tmp_avg)
                child_avgs.append(inner_temp)
                


            # child_grid1 = grid_copy[0:child_grid_h, 0:child_grid_w ].copy() * grid_avg
            # child_grid1_avg = np.mean(child_grid1)
            # child_grid2 = grid_copy[child_grid_h:, 0:child_grid_w ].copy() * grid_avg
            # child_grid2_avg = np.mean(child_grid2)
            # child_grid3 = grid_copy[child_grid_h:,child_grid_w: ].copy() * grid_avg
            # child_grid3_avg = np.mean(child_grid3)
            # child_grid4 = grid_copy[0:child_grid_h:,child_grid_w: ].copy() * grid_avg
            # child_grid4_avg = np.mean(child_grid4)

            # treshold = 60
            # # child_grid1_avg*=grid_avg/255
            # # child_grid2_avg*=grid_avg/255
            # # child_grid3_avg*=grid_avg/255
            # # child_grid4_avg*=grid_avg/255

            # child_grid1_avg = 0 if child_grid1_avg < 60 else  255
            # child_grid2_avg = 0 if child_grid2_avg < 60 else  255
            # child_grid3_avg = 0 if child_grid3_avg < 60 else  255
            # child_grid4_avg = 0 if child_grid4_avg < 60 else  255

            

            for i_1 in range(len(child_avgs)):
                for j_1 in range(len(child_avgs[0])):
                    thresh1[i+ (i_1*child_grid_h):i+ ((i_1+1)*child_grid_h),
                            j+ (j_1*child_grid_h):j+ ((j_1+1)*child_grid_w)] = child_avgs[i_1][j_1] #if child_avgs[i_1][j_1] # < 10 else 255
            
            # thresh1[i+child_grid_h:i+child_grid_h+child_grid_h,j:j+child_grid_w] = child_grid2_avg
            # thresh1[i+child_grid_h:i+child_grid_h+child_grid_h,j+child_grid_w:j+child_grid_w+child_grid_w] = child_grid3_avg 
            # thresh1[i:i+child_grid_h,j+child_grid_w:j+child_grid_w+child_grid_w] = child_grid4_avg
            




    # Taking a matrix of size 5 as the kernel
    # kernel = np.ones((150, 150), np.uint8)
    
    # # The first parameter is the original image,
    # # kernel is the matrix with which image is
    # # convolved and third parameter is the number
    # # of iterations, which will determine how much
    # # you want to erode/dilate a given image.
    # img_dilation = cv.dilate(thresh1, kernel, iterations=1)
    # img_erosion = cv.erode(img_dilation, kernel, iterations=1)
    """
    """

    #cv.imshow("Ortalamayla carpma", thresh1)cv.imshow("After Tresholding Sobel Operation", thresh1_v1)
    

    cv.waitKey(0)
    cv.destroyAllWindows()
    # Combine the two images to get the foreground.
    #im_out = thresh1 | im_floodfill_inv
    
    
    # # Taking a matrix of size 5 as the kernel
    # kernel_e = np.ones((3, 3), np.uint8)
    
    # # The first parameter is the original image,
    # # kernel is the matrix with which image is
    # # convolved and third parameter is the number
    # # of iterations, which will determine how much
    # # you want to erode/dilate a given image.
    # img_erosion = cv.erode(thresh1, kernel_e, iterations=1)
    
    # kernel_d = np.ones((3, 3), np.uint8)
    # img_erosion = cv.dilate(img_erosion, kernel_d, iterations=1)



    
 

    cv.waitKey(0)
    plt.axis('off')
    plt.title("Virtual Map")
    plt.show()
    
    
    return 0
if __name__ == "__main__":
    main(sys.argv[1:])