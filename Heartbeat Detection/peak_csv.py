import os
import cv2
import numpy as np
import csv

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    print(labels[:, :][labels>0])
    print(type(labels))
    print('max:',np.amax(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display    
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[labels==0] = 0   
    # 除雜訊
    # max_lable = np.amax(labels)
    # labeled_img[labels<max_lable] = 0  
    data.append(np.amax(labels)*6)
    with open('heartbeat.csv', 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)            
        writer.writerow(data)   
    #cv2.imshow('labeled.png', labeled_img)


# get the path/directory
folder_dir = 'EKG'
# folder_dir = "/mnt/c/Users/R6/Desktop/rgbd" #ubuntu
for filename in os.listdir(folder_dir):
 
    # check if the image ends with png
    if (filename.endswith(".jpg")):
        img = cv2.imread(os.path.join(folder_dir,filename))
        csv_filename = os.path.splitext(folder_dir+'_'+filename)[0]
        print(csv_filename)
        # print(img.shape)
        data = [csv_filename]
        
    
        # img = cv2.imread('EKG_001-120/15.jpg')
        crop_img = img[805:950, 114:1340]
        # cv2.imshow('crop_img', crop_img)
        blur = cv2.GaussianBlur(crop_img, (5,5), 0)
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0,0,0])
        upper_black = np.array([200,200,200])
        mask = cv2.inRange(hsv, lower_black, upper_black)
        # cv2.imshow('mask', mask)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        erode_mask = cv2.erode(mask, kernel, iterations = 1)
        
        #找最大連通域面積 CC_STAT_AREA
        nb_components, labels, stats, centroids2 = cv2.connectedComponentsWithStats(erode_mask, connectivity=4)
        max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1]) 
        print(max_size)
        img2 = np.zeros(labels.shape)
        img2[labels == max_label] = 255
        img2 = np.uint8(img2)
        # cv2.imshow("Biggest component", img2)
        
        
        erode_kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        img2_erode_mask = cv2.erode(img2, erode_kernel2, iterations = 2)
        # cv2.imshow('erode_mask2', img2_erode_mask)
        
        
        img3_components, output, stats3, centroids3 = cv2.connectedComponentsWithStats(img2_erode_mask, connectivity=4)
        print(stats3[:,0])
        print(stats3[:,1])
        sum_size = sum([(stats3[i, cv2.CC_STAT_AREA]) for i in range(1, img3_components)]) 
        avg_size = sum_size/(img3_components + 1)
        print('stats',stats3[:,4])
        img3_lst = []
        img3 = np.zeros(output.shape)
        for i in range(1, img3_components):
            if stats3[i][4] > 3:
                img3_lst.append(i)
                img3[output == i] = 255
        img3 = np.uint8(img3)
        
        dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilation = cv2.dilate(img3, dilation_kernel, iterations = 6)
        
        num_labels, labels_im = cv2.connectedComponents(dilation)
        imshow_components(labels_im)
                


# cv2.waitKey(0)
# cv2.destroyAllWindows()