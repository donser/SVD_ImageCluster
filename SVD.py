'''
Edit by donser
SVD_ImageCluster
SVD for image compressing using：python/opencv
'''
import cv2 as cv
import numpy as np
from sys import argv
#输入图像名 输出图像名 选取的奇异值数目
py , input_name , output_name , singular_num= argv
N = int(singular_num)

'''
输入统一为三通道
'''
img=cv.imread(input_name)
try:
    img.shape
except:
    print("Faile to load:",input_name)
    sys.exit(0)
img = img.astype(np.float32)
h = img.shape[0]
w = img.shape[1]
c = img.shape[2]

'''
矩阵分解 img=U*Sigma*VT
'''
U=list()
Sigma=list()
VT=list()
for i in range(3):
    U_tmp,Sigma_tmp,VT_tmp = np.linalg.svd(img[:,:,i])
    U.append(U_tmp)
    Sigma.append(Sigma_tmp)
    VT.append(VT_tmp)

'''
类型转换 list->array
'''
U=np.array(U)
Sigma=np.array(Sigma)
VT=np.array(VT)

'''
计算压缩比率 = h*w / (singular_num*(h+w+1))
'''
print( "Compression Ratio:" , h*w/(N*(h+w+1)) )

'''
根据奇异值的数目，组合输出数组
'''
img_out=np.zeros(img.shape)

for i in range (3):
    FULL_Sigma=np.diag(Sigma[i,:N])
    tmp=np.dot(U[i,:,:N],FULL_Sigma).dot(VT[i,:N,:])
    img_out[:,:,i]=tmp
#特值处理
img_out[img_out<0]=0
img_out[img_out>255]=255
img_out = img_out.astype(np.uint8)
#绘图
cv.imwrite(output_name,img_out)


