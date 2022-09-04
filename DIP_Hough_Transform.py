"""
DIP Lab Mini Project

Rahul Kumar Meena   --  18EC35023
Shreya Kumari       --  18EC35027
Sn Ramanathan       --  18EC35028

TA- Soumyadeep Kal
"""

import numpy as np
import cv2 as cv
import math 
import matplotlib.pyplot as plt 
from tqdm import tqdm
import keyboard

sobel_diagnol3 =             [[-1.0, -0.5, 0.0],
                              [-0.5,  0.0, 0.5],
                              [ 0.0,  0.5, 1.0]]

sobel_diagnol5 =              [[-0.50, -0.4, -0.25, -0.2, 0.0],
                               [-0.40, -1.0, -0.50, 0.0, 0.2],
                               [-0.25, -0.5,  0.00, 0.5, 0.25],
                               [-0.20,  0.0,  0.50, 1.0, 0.4],
                               [ 0.00,  0.2,  0.25, 0.4, 0.5]]

sobel_diagnol7 =                [[-0.333333, -0.3, -0.230769, -0.166667, -0.153846, -0.1, 0.0],
                                [-0.3, -0.50, -0.4, -0.25, -0.2, 0, 0.1],
                                [-0.230769, -0.40, -1, -0.5, 0.0, 0.2, 0.153846],
                                [-0.166667, -0.25, -0.5, 0, 0.5, 0.25, 0.166667],
                                [-0.153846, -0.2, 0.0, 0.5, 1, 0.4, 0.230769 ],
                                [-0.1, 0.0, 0.2, 0.25, 0.4, 0.5, 0.3],
                                [0, 0.1, 0.153846, 0.166667, 0.230769, 0.3, 0.333333]]

sobel_diagnol9 =                [[-0.25,-0.235294,-0.2,-0.16,-0.125,-0.12,-0.1,-0.0588235,0],
                                [-0.235294,-0.333333, -0.3, -0.230769, -0.166667, -0.153846, -0.1, 0.0,0.1],
                                [-0.2,-0.3, -0.50, -0.4, -0.25, -0.2, 0, 0.1,0.1],
                                [-0.16,-0.230769, -0.40, -1, -0.5, 0.0, 0.2, 0.153846,0.12],
                                [-0.125,-0.166667, -0.25, -0.5, 0, 0.5, 0.25, 0.166667,0.125],
                                [-0.12,-0.153846, -0.2, 0.0, 0.5, 1, 0.4, 0.230769, 0.16],
                                [-0.1,-0.1, 0.0, 0.2, 0.25, 0.4, 0.5, 0.3,0.2],
                                [-0.0588235,0, 0.1, 0.153846, 0.166667, 0.230769, 0.3, 0.333333,0.235294],
                                [0,0.0588235,0.1,0.12,0.125,0.16,0.2,0.235294,0.25]]

def sobel_diagnol_filter_coff(f_size, x, y):

    if (f_size == 3):
        return sobel_diagnol3[x][y];
    elif (f_size == 5):
        return sobel_diagnol5[x][y];
    elif (f_size == 7):
        return sobel_diagnol7[x][y];
    else:
        return sobel_diagnol9[x][y];

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)

def pixel_value(i,j):
    val=0
    if((i >= 0 and i < img_rows) and (j >= 0 and j < img_cols)):
        val = img[i][j]
    return val

def Sobel_diagonal_filter(img,f_size):
    
    img_rows,img_cols=img.shape
    out_img = np.zeros((img_rows,img_cols,3)).astype(np.uint8)
    
    for i in tqdm(range(img_rows),desc="Loading..."):
        for j in range(img_cols):  
            sum_x = 0 
            sum_y=0
            temp_x=0
            temp_y=0
            norm_fact_x=0
            norm_fact_y=0
            for k in range(-1 * math.floor((f_size - 1)/2) , math.floor((f_size - 1)/2)+1):
                for l in range(-1 * math.floor((f_size - 1)/2) , math.floor((f_size - 1) / 2)+1):
                    temp_x = sobel_diagnol_filter_coff(f_size,int(k + math.floor(f_size - 1) / 2),int( l+ math.floor((f_size - 1) / 2) ))
                    temp_y = sobel_diagnol_filter_coff (f_size,int(k + math.floor(f_size - 1) / 2),int(math.floor((f_size - 1) / 2)-l ))
                    sum_x += temp_x * pixel_value(i + k, j + l)
                    sum_y += temp_y * pixel_value(i + k, j + l)
                    
                    if (temp_x > 0):
                        norm_fact_x += temp_x
                    if (temp_y > 0):
                        norm_fact_y += temp_y
            out_img[i][j][0]=int(round(math.sqrt((sum_x / norm_fact_x) * (sum_x / norm_fact_x) + (sum_y / norm_fact_y) * (sum_y / norm_fact_y))))
            if(out_img[i][j][0]>255):
                out_img[i][j][0]=255
            out_img[i][j][1] =out_img[i][j][0]
            out_img[i][j][2]=out_img[i][j][0]

    return out_img


def hough_line(img, value_threshold=5, accum_thresh=150,angle_step=10, lines_are_white=True ):
   
    # Rho and Theta ranges
    thetas = np.deg2rad(np.arange(-90.0, 90.0, angle_step))
    width, height = img.shape
    diag_len = int(round(math.sqrt(width * width + height * height)))
    rhos = np.linspace(-diag_len, diag_len, diag_len * 2)

    # Cache some resuable values
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)

    # Hough accumulator array of theta vs rho
    accumulator = np.zeros((2 * diag_len, num_thetas), dtype=np.uint8)
    # (row, col) indexes to edges
    are_edges = img > value_threshold if lines_are_white else img < value_threshold
    y_idxs, x_idxs = np.nonzero(are_edges)

    # Vote in the hough accumulator
    for i in tqdm(range(len(x_idxs)),desc="Loading..."):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            accumulator[rho, t_idx] += 1

    out_img =np.zeros((width,height,3), dtype=np.uint8)
    for i in tqdm(range(len(x_idxs)),desc="Loading..."):
        x = x_idxs[i]
        y = y_idxs[i]
        is_line = False
        for t_idx in range(num_thetas):
            # Calculate rho. diag_len is added for a positive index
            rho = diag_len + int(round(x * cos_t[t_idx] + y * sin_t[t_idx]))
            if(accumulator[rho, t_idx] > accum_thresh):
                is_line = True
            
        out_img[y][x][0]=is_line*255
        out_img[y][x][1]=is_line*255
        out_img[y][x][2]=is_line*255


    return accumulator, thetas, rhos, out_img,are_edges


def show_hough_line(img,out_img, are_edges,accumulator, thetas, rhos, save_path=None,save_path1=None):
    
  
    fig, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(img, cmap=plt.cm.gray)
    ax[0].set_title('Input image')
    ax[0].axis('image')
  
  
    c=ax[1].imshow(
        accumulator, cmap='jet',
        extent=[np.rad2deg(thetas[-1]), np.rad2deg(thetas[0]), rhos[-1], rhos[0]])
    ax[1].set_aspect(1/5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    plt.colorbar(c)
  
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    # plt.show()


    fig2, ax1 = plt.subplots(1, 2, figsize=(10, 10))
    ax1[1].imshow(out_img, cmap=plt.cm.gray)
    ax1[1].set_title('Output image')
    ax1[0].imshow(are_edges, cmap=plt.cm.gray)
    ax1[0].set_title('Binarized image')
    if save_path1 is not None:
        plt.savefig(save_path1, bbox_inches='tight')

    # ax1[0].axis('image')
    plt.show()
    return img
   

def hough(val,val1,val2):
    accumulator, thetas, rhos, out_img,are_edges = hough_line(img,val,val1,val2)
    cv.imshow('output',out_img)
    show_hough_line(img,out_img,are_edges, accumulator, thetas, rhos, save_path='Hough_transform.png',save_path1='Output.png')

def null(x):
    pass


if __name__ == '__main__':
    print("Do you want to do Edge Detection")
    print("Press Y or N\n")
    while True:
        try:  # used try so that if user pressed other than the given key error will not be shown
            if keyboard.is_pressed('y'):  # if key 'q' is pressed 
                print('You Pressed Y')
                val = input("Enter Image Name for example lena_gray_512.jpg\n")
                img=cv.imread(val)
                if img.ndim == 3:
                    img = rgb2gray(img)
                img_rows,img_cols = img.shape
                val = input("\nEnter Filter_size: 3 or 5 or 7 or 9\n")
                f_size=int(val)
                out_img_edge=Sobel_diagonal_filter(img,f_size)
                cv.imwrite('output_edge.jpg', out_img_edge)     
                cv.imshow('Edge Detected Image',out_img_edge)      
                cv.waitKey(0)
                break  # finishing the loop
            if keyboard.is_pressed('n'):
                break
        except:
            break  # if user pressed a key other than the given key the loop will break
    cv.destroyAllWindows()
    val = input("Enter Image Name for example output_edge.jpg\n")  
    img = cv.imread(val)
    if(img is not None):
        if img.ndim == 3:
            img = rgb2gray(img)
        
        cv.namedWindow('output')    
        cv.createTrackbar('Img_thresh','output',0,255,null)
        cv.createTrackbar('Accu_thresh','output',1,255,null)
        cv.createTrackbar('angle_step','output',1,255,null)
        print("\n\n\ntrackbar 1 : Image Binarization Threshold")
        print("trackbar 2 : Accumulator Threshold Threshold")
        print("trackbar 3 : Angle Quantization Steps\n\n")
        print("Press P to do hough transform and Q to quit" )
        print("To do Hough transform again just close both figures, choose desired controls and Press P")
        # hough(60,250,5)
        while (True):
            key = cv.waitKey(1) & 0xFF
            if keyboard.is_pressed('p'):
                pos = cv.getTrackbarPos('Img_thresh', 'output')
                pos1 = cv.getTrackbarPos('Accu_thresh', 'output')
                pos2 = cv.getTrackbarPos('angle_step', 'output')
                hough(pos,pos1,pos2)
            if keyboard.is_pressed('q'):
                break
        cv.destroyAllWindows()
    else: 
        print("Please do edge detection first.")
                

       
    