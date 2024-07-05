'''https://blog.gtwang.org/programming/how-to-crop-an-image-in-opencv-using-python/
將每張ECG圖存成12張txt
'''
import cv2
import sys
import numpy as np

# 讀取圖檔
img = cv2.imread("EKG_481-600/590.jpg")

# 裁切區域的 x 與 y 座標（左上角）
x = 119
y = 390

# 裁切區域的長度與寬度
w = 306
h = 141

# 裁切圖片
# ekg_list = ['l', 'll', 'lll', 'aVR', 'aVL', 'avF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
# crop_img = []
# for x in range(119,1340,306):
#     if x == 731:
#         x = x-2
#     if x == 1037:
#         x = x-2        
#     if x == 425:
#         w = 304
#     else:
#         w = 306
#     for y in range(390,813,141):
#         crop_img.append(img[y:y+h, x:x+w])      
        # print(x,y)    
# np.set_printoptions(threshold=sys.maxsize)
# print(crop_img[0])

# 存成12張txt
# for i in range(12):#        
#     crop_name = 'EKG_590_' + ekg_list[i] + '.txt' #str(j)
#     with open(crop_name, "w") as file:
#         line = ''.join(''.join(map(str, l)) for l in crop_img[i])
#         file.write(line)

# j=0
# for i in range(len((crop_img[j]))):
#     for j in range(len(crop_img)):        
#             for k in range(len((crop_img[j][i]))):  
#                 print(crop_img[j][i][k],end = ' ')              
# print(crop_img[11][140][302])
# for l in crop_img[1]:
#     print(*[e[0] for e in l])
# print (str(crop_img[1])[1:-1] )

# 依序顯示12張圖片 
# for i in range(12):#        
#     cv2.imshow("cropped", crop_img[i])
#     cv2.waitKey(0)
# cv2.destroyAllWindows()   

# 畫12個框框
cv2.rectangle(img, (119,390), (425, 531), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (119, 531), (425, 672), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (119, 672), (425, 813), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (425, 390), (729, 531), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (425, 531), (729, 672), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (425, 672), (729, 813), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (729, 390), (1035, 531), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (729, 531), (1035, 672), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (729, 672), (1035, 813), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (1035, 390), (1340, 531), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (1035, 531), (1340, 672), (0, 0, 255), 1, cv2.LINE_AA)
cv2.rectangle(img, (1035, 672), (1340, 813), (0, 0, 255), 1, cv2.LINE_AA)
crop_img = img[390:814, 119:1341]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("EKG_590.jpg", crop_img)
