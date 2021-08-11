import cv2
import glob
import numpy as np

def readCameraMatrixs():
    mtx = np.array(
        [[1.14647559e+03, 0.00000000e+00,6.42472076e+02],
        [0.00000000e+00, 1.14521026e+03, 4.69462953e+02],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]],dtype=np.float32
    )

    dist = np.array(
        [[-2.56813095e-01,  1.37128239e-01, -1.83450939e-04,  2.55137252e-04,-7.32416009e-02]],dtype=np.float32
    )

    newcameramtx = np.array(
        [[996.74487305,   0.,         643.06154572],
        [  0.,         996.87188721, 467.41649679],
        [  0. ,          0. ,          1.        ]],dtype=np.float32
    )
    # return None # 重新做内参标定就返回空
    return mtx,dist,newcameramtx

def calibrate():
    internal_param = readCameraMatrixs()
    if internal_param is None:
        criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
        objp = np.zeros((7 * 4, 3), np.float32)
        objp[:, :2] = np.mgrid[0:4, 0:7].T.reshape(-1, 2)  # 获取标定板角点的位置， 我们先标定内参，因此无需关注世界坐标系的位置。按内角点的顺序从0-4,0-7即可

        obj_points = []
        img_points = []
        images = glob.glob(".\\calibrationPicture\\*.jpg")
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            ret, corners = cv2.findChessboardCorners(gray, (4, 7), None)
            if ret:
                obj_points.append(objp)
                corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
                if [corners2]:
                    img_points.append(corners2)
                else:
                    img_points.append(corners)

                cv2.drawChessboardCorners(img, (4, 7), corners, ret)  # 显示角点顺序
                # cv2.imshow('findCorners',img)
                cv2.waitKey(10)
        cv2.destroyAllWindows()

        img = cv2.imread("test.jpg")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None) # 返回内参矩阵mtx,畸变系数dist,旋转向量rvecs,平移矩阵tvecs。由于obj_points没有输入真实的世界坐标，因此rvecs和tvecs不作参考
        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))#获取畸变矫正outer newcameramtx，alpha参数设为1，保留所有像素点，和矫正前图片尺寸相同，但是会有黑边
        dst = cv2.undistort(img,mtx,dist,None,newcameramtx)#进行畸变矫正
        x,y,w,h = roi # 矫正后的内矩阵区域，也就是去掉黑边的区域。参考：https://www.cnblogs.com/riddick/p/6711263.html
        # dst1 = dst[y:y+h,x:x+w] # 裁剪矫正后的黑边
        # dst1 = cv2.resize(dst1,(1280,960))
        cv2.imwrite('calibrate_alpha1_2.jpg', dst)
    else:
        mtx,dist,newcameramtx = internal_param
    # TODO 保存mtx和dist，完成内参标定
    print("畸变内参矩阵mtx:\n", mtx)  
    print("无畸变内参矩阵newcameramtx:\n", newcameramtx)  
    print("畸变系数dist:\n", dist ) 
    return mtx,dist,newcameramtx



def calcRT(obj_coordinate,img_coordinate,mtx,dist):
    size = (1280,960) # 这里根据实际图像尺寸修改
    ret,rvecs, tvecs  = cv2.solvePnP(obj_coordinate,img_coordinate,mtx,dist) 
    # 这里注意：如果内参矩阵传的是outer newcameramtx，则img_coordinate为矫正后的坐标。如果传mtx，则为矫正前原图的坐标。

    #也可以使用calibrateCamera，但是要注意传参，坐标点需要放在数组中，并设置flags为CALIB_USE_INTRINSIC_GUESS，返回结果也是一个数组，和参数数组顺序对应
    #obj_points = []
    #img_points = []
    #obj_points.append(obj_coordinate)
    #img_points.append(img_coordinate)
    #ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, mtx, dist,flags=cv2.CALIB_USE_INTRINSIC_GUESS)

    R = cv2.Rodrigues(rvecs)[0]
    T = tvecs
    print("旋转矩阵R:\n",R) 
    print("平移矩阵T:\n", T) 
    return R,T,rvecs # 返回旋转矩阵、平移矩阵、旋转向量


def calcPW(u,v,z_c,mtx,dist,R,T):
    undist_point = cv2.undistortPoints(np.array([[u,v]],dtype=np.float32),mtx,dist)
    ## 测试矫正坐标点
    print("opencv矫正坐标点：",undist_point)
    x_c = undist_point[0][0][0]
    y_c = undist_point[0][0][1]
    calc_point = (u - mtx[0][2]) / mtx[0][0],(v - mtx[1][2]) / mtx[1][1]
    print("计算矫正坐标点：",calc_point)
    # x_c = calc_point[0]
    # y_c = calc_point[1]
    # x = R[0][0] * ((u - mtx[0][2]) * z_c / mtx[0][0] - T[0]) + R[1][0] * ((v - mtx[1][2]) * z_c / mtx[1][1] - T[1]) + R[2][0] * (z_c - T[2])
    # y = R[0][1] * ((u - mtx[0][2]) * z_c / mtx[0][0] - T[0]) + R[1][1] * ((v - mtx[1][2]) * z_c / mtx[1][1] - T[1]) + R[2][1] * (z_c - T[2])
    # z = R[0][2] * ((u - mtx[0][2]) * z_c / mtx[0][0] - T[0]) + R[1][2] * ((v - mtx[1][2]) * z_c / mtx[1][1] - T[1]) + R[2][2] * (z_c - T[2])
    x = R[0][0] * (x_c * z_c  - T[0]) + R[1][0] * (y_c * z_c  - T[1]) + R[2][0] * (z_c - T[2])
    y = R[0][1] * (x_c * z_c  - T[0]) + R[1][1] * (y_c * z_c  - T[1]) + R[2][1] * (z_c - T[2])
    z = R[0][2] * (x_c * z_c  - T[0]) + R[1][2] * (y_c * z_c  - T[1]) + R[2][2] * (z_c - T[2])
    return x,y,z

def reproject(obj_eval,img_eval,rvecs,T,mtx,dist):
    print('########## 评估反投影误差 ##########')
    imgpoints_reflect, _ = cv2.projectPoints(obj_eval, rvecs, T, mtx, dist)
    print("反投影点像素坐标：\n",imgpoints_reflect)
    print("实际像素坐标：\n",img_eval)
    error = cv2.norm(img_eval,imgpoints_reflect, cv2.NORM_L2)/len(imgpoints_reflect)
    print("反投影误差error: ",error)


if __name__ == "__main__":
    print('########## 一、开始内参标定 ##########')
    mtx,dist,newcameramtx = calibrate()
    print('########## 二、开始外参标定 ##########')
    obj_coordinate = np.array(
        [
            [0,0,0],
            [500,0,0],
            [1000,0,0],
            [1000,500,0],
            [1000,1000,0],
            [0,1000,0],
        ],dtype=np.float32
    )# 机器人坐标，3D标定需要至少6个坐标，2D标定也就是所有点的z坐标相同，需要至少4个坐标

    img_coordinate = np.array(
        [
            # alpha = 1 outer newcameramtx矫正矫正
            [305,827],
            [303,583],
            [302,335],
            [548,333],
            [800,332],
            [801,832],
        ],dtype=np.float32
    )# 像素坐标，3D标定需要至少6个坐标，2D标定也就是所有点的z坐标相同，需要至少4个坐标
    R,T,rvecs = calcRT(obj_coordinate,img_coordinate,newcameramtx,None)  
    
    u = 300 # 测试点，像素坐标x
    v = 85   # 测试点，像素坐标y
    z_c = 2000  # 相机离目标物体的垂直距离
    x,y,z = calcPW(u,v,z_c,newcameramtx,None,R,T)
    print("像素坐标：\n({},{})".format(u,v))
    print("机器人坐标x:",x)
    print("机器人坐标y:",y)
    print("机器人坐标z:",z)
    print('########## 所有标定完成 ##########')


    # 用另外的坐标点评估
    obj_eval = np.array(
        [
            [[0,1500,0]],
        ],dtype=np.float32
    )# 机器人坐标,反投影误差需要用三维数组表示坐标

    img_eval = np.array(
        [
            # alpha = 0 inner newcameramtx矫正
            # [[1071,827]],
            # 未矫正
            [[1088,856]],
            # alpha = 1 outer newcameramtx矫正
            # [[1043,796]],
        ],dtype=np.float32
    )# 像素坐标,反投影误差需要用三维数组表示坐标
    reproject(obj_eval,img_eval,rvecs,T,mtx,dist)
