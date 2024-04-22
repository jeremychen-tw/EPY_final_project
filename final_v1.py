from time import sleep
from PyQt5.QtWidgets import *
from PyQt5 import uic
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import cv2
from matplotlib import pyplot as plt
import numpy as np
import os
import math
import glob
import copy
from numpy.linalg import inv
import time



class MyGUI(QMainWindow):
    def __init__(self):
        global imageSz
        global chessBoardSize
        global gridBlockSideL
        super(MyGUI,self).__init__()
        uic.loadUi("final_project.ui", self)
        self.show()
        gridBlockSideL=1.9 # length of block on the chessboard in mm
        imageSz = (480,640)
        chessBoardSize = (6,9) # the grid size of the chessboard
        
        # setting up button
        self.pushButton_IC.clicked.connect(self.cameraInit)
        self.pushButton_CC.clicked.connect(self.cameraCalibration)
        self.pushButton_DLT.clicked.connect(self.drawLineTest)
        self.pushButton_OT.clicked.connect(self.objectTracking)
        self.pushButton_PP.clicked.connect(self.plotPath)
        self.pushButton_RP.clicked.connect(self.reverseProj)
        self.pushButton_AP.clicked.connect(self.pathAnalysis)
        self.pushButton_SFCSV.clicked.connect(self.savePath2CSV)
        self.pushButton_SP.clicked.connect(self.smoothenPath)
    
    # Initiate the camera to read in a image for calibration
    def cameraInit(self):
        self.statusText.setText("Initiating Camera...")
        global frame
        vid = cv2.VideoCapture(0) 
        frame = np.zeros(1)
        
        while(True): 
            # Capture the video frame by frame 
            ret, frame = vid.read() 
        
            # Display the resulting frame 
            cv2.imshow('frame (Press Q to select the frame)', frame) 
            
            # the 'q' button is set as the quitting button you may use any desired button of your choice 
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
        
        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 
        self.statusText.setText("Camera initiated!")

    # Calibrate camera
    def cameraCalibration(self):
        global dist
        global rotate
        global trans
        global intrinsicMat
        self.statusText.setText("Calibrating Camera...")
        # finding corners on the chessboard
        image = copy.deepcopy(frame) # deep copy to avoid changing the original
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # creating the grid loaction of the board, here we do not have 
        # exact size so we can create simple grid like(1,0)(1,1)(1,2)....
        objp = np.zeros((chessBoardSize[0]*chessBoardSize[1],3), np.float32)
        objp[:,:2] = np.mgrid[0:chessBoardSize[0],0:chessBoardSize[1]].T.reshape(-1,2)*gridBlockSideL

        # the objpoints are the2D points on the chessboard that will have a correspondin 3D location in object point
        objpoints = [] 
        imgpoints = []

        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        retval, corners = cv2.findChessboardCorners(image, chessBoardSize, None)
        image_with_lines = cv2.drawChessboardCorners(image, chessBoardSize, corners, retval) # use image to get colored line but not gray

        #show it corner finding is successful
        if retval:
            self.statusText.setText("Corner found!\nCalibration Done!")
            cv2.imshow('image', image_with_lines)
            cv2.waitKey(10000)
            cv2.destroyAllWindows()
        else:
            self.statusText.setText("Corner not found")
            
        # adding the 2D points on the chessboard and adding the corresponding 3D points
        objpoints.append(objp)
        cornersToAppend = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(cornersToAppend)

        # using the camera params to calibrate footage to avoid warpping
        retval, intrinsicMat, dist, rotate, trans = cv2.calibrateCamera(objpoints, imgpoints, imageSz,None,None)

        newcameramtx, roi=cv2.getOptimalNewCameraMatrix(intrinsicMat,dist,(imageSz[1],imageSz[0]),1,(imageSz[1],imageSz[0]))
        # undistort
        dst = cv2.undistort(image, intrinsicMat, dist, None, newcameramtx)
        
    
    def drawLineTest(self):
        image = copy.deepcopy(frame)

        imgpts, jacobian = cv2.projectPoints(np.array([[0,0,0],[gridBlockSideL*(chessBoardSize[0]-1),gridBlockSideL*(chessBoardSize[1]-1),0]]), rotate[0], trans[0], intrinsicMat, dist)
        image = cv2.line(image, tuple((imgpts[0].ravel()).astype(int)), tuple((imgpts[1].ravel()).astype(int)), (0, 0, 255), 3)

        cv2.imshow('Draw Line Test',image)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()


    def objectTracking(self):
        self.statusText.setText("Gathering Data...")
        global centroids
        # Define range for red color in HSV
        lower_red1 = np.array([0, 150, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 150, 100])
        upper_red2 = np.array([180, 255, 255])

        vid = cv2.VideoCapture(0) 
        frame = np.zeros(1)

        centroids = []

        # create a timer to record the time
        start_time = time.time()

        # get a first framw to use as background
        ret, firstframe = vid.read() 

        while(True): 
            # Capture the video frame by frame 
            ret, frame = vid.read() 
        
            # convert to HSV for better color detection
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Threshold the HSV image to get only red colors
            mask1 = cv2.inRange(image, lower_red1, upper_red1)
            mask2 = cv2.inRange(image, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours in the mask
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if any contour was found
            if contours:
                contour = max(contours, key=cv2.contourArea)

                # Calculate moments for the contour
                M = cv2.moments(contour)
                # Calculate centroid
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    elapsed_time = time.time() - start_time
                    centroid = (elapsed_time,cX, cY)
                    centroids.append(centroid)

                    # Draw a green dot at the centroid
                    cv2.circle(firstframe, centroid[1:], 3, (0, 255, 0), -1)
                    # Draw a green dot at the centroid
                    cv2.circle(frame, centroid[1:], 3, (0, 255, 0), -1)

            # Show the resulting image with green dot at the centroid
            cv2.imshow('Path record', firstframe)
            cv2.imshow('Camera Feed', frame)
            cv2.moveWindow('Path record', 0, 0)
            cv2.moveWindow('Camera Feed', firstframe.shape[1], 0)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                cv2.destroyAllWindows()
                break

        vid.release()
        cv2.imshow('Path record', firstframe)
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        self.statusText.setText("End of Data Collection!")


    def plotPath(self):
        self.statusText.setText("Plotting Path...")
        # Assuming 'centroids' is your list of tuples
        _ ,x_coords, y_coords = zip(*centroids)

        plt.plot(x_coords, y_coords, 'g-')
        plt.gca().invert_yaxis() 
        plt.show()
        self.statusText.setText("Path plotted!")
        
    def smoothenPath(self):
        self.statusText.setText("Smoothening Path...")
        global centroids
        # Define the size of the moving average window
        window_size = 5
        time ,x_coords, y_coords = zip(*centroids)
        # Calculate the moving averages
        x_coords_smooth = np.convolve(x_coords, np.ones(window_size)/window_size, mode='valid')
        y_coords_smooth = np.convolve(y_coords, np.ones(window_size)/window_size, mode='valid')
        
        # Save the smoothed data back to centroids
        centroids_smooth = list(zip(time, x_coords_smooth, y_coords_smooth))
        centroids = centroids_smooth
        self.statusText.setText("Path Smoothened!")
        
        
    def reverseProj(self):
        self.statusText.setText("Performing Reverse Projection...")
        rotMat = cv2.Rodrigues(rotate[0])[0]
        # print("Rotmat=\n", rotMat, "\n")
        # print("trans=\n", trans, "\n")
        extrinsicMat = np.hstack((rotMat,-trans[0]))
        # print("extrinsicMat=\n", extrinsicMat, "\n")
        projMat = intrinsicMat.dot(extrinsicMat)
        # print("projMat=\n", projMat, "\n")
        centroidsNp = np.array(centroids)
        centroidsNp = np.hstack((centroidsNp, np.ones((centroidsNp.shape[0],1))))
        leftSideMat  = inv(rotMat).dot(inv(intrinsicMat)).dot(centroidsNp.T[1:])
        rightSideMat = inv(rotMat).dot(trans[0]).reshape(3,1)
        s = (0 + rightSideMat[2])/leftSideMat[2]
        ptChessBd = inv(rotMat).dot(s * inv(intrinsicMat).dot(centroidsNp.T[1:]) - trans[0])
        self.statusText.setText("Reverse Projection Done!")


    def pathAnalysis(self):
        self.statusText.setText("Analyzing Path...")
        global speed
        global acceleration
        global curvature
        # Assuming 'centroids' is your list of tuples
        time, x_coords, y_coords = zip(*centroids)

        window_size = 5
        
        # Convert to numpy arrays for easier calculations
        time = np.array(time)
        x_coords = np.array(x_coords[:-(window_size-1)])
        y_coords = np.array(y_coords[:-(window_size-1)])

        # Trim the time array to match the length of the smoothed arrays
        time = time[:-(window_size-1)]

        # differentiate
        dt = np.diff(time)
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)

        # speed
        speed = np.sqrt(dx**2 + dy**2) / dt

        # acceleration
        acceleration = np.diff(speed) / dt[:-1]

        # curvature
        epsilon = 1e-5  # small constant to avoid division by zero
        curvature = np.abs(dx[:-1] * np.diff(dy) - np.diff(dx) * dy[:-1]) / (dx[:-1]**2 + dy[:-1]**2 + epsilon)**1.5

        self.statusText.setText("Plotting Analysis...")
        # Path for reference
        plt.subplot(2, 2, 1)
        plt.plot(x_coords, y_coords, 'g-')
        plt.gca().invert_yaxis()
        plt.title('Speed vs Time')

        # Speed plot
        plt.subplot(2, 2, 2)
        plt.plot(time[:-1], speed)
        plt.title('Speed vs Time')
        plt.xlabel('Time')
        plt.ylabel('Speed')

        # Acceleration plot
        plt.subplot(2, 2, 3)
        plt.plot(time[:-2], acceleration)
        plt.title('Acceleration vs Time')
        plt.xlabel('Time')
        plt.ylabel('Acceleration')

        # Curvature Plot
        plt.subplot(2, 2, 4)
        plt.plot(time[:-2], curvature)
        plt.title('Curvature vs Time')
        plt.xlabel('Time')
        plt.ylabel('Curvature')

        plt.show()
        
    def savePath2CSV(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File") 
        if not filePath.endswith('.csv'):
            filePath += '.csv'
            
        # Assuming 'centroids' is your list of tuples
        time, x_coords, y_coords = zip(*centroids)
        # Save the path to a CSV file
        with open(filePath, 'w') as f:
            f.write('Time,X,Y,Speed,Acceleration,Curvature\n')
            for i in range(len(time)):
                speed_value = speed[i] if i < len(speed) else ''
                acceleration_value = acceleration[i] if i < len(acceleration) else ''
                curvature_value = curvature[i] if i < len(curvature) else ''
                f.write(f'{time[i]},{x_coords[i]},{y_coords[i]},{speed_value},{acceleration_value},{curvature_value}\n')
            f.close()
        self.savedRes.setText("Path saved to\n" + filePath)


def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()


if __name__ == "__main__":
    main()
