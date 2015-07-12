import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from pandas import read_csv
from PIL import Image, ImageDraw

train = read_csv('d:/Downloads/train.csv')
trainX = np.asarray(train[range(0,3)])
trainY = np.asarray(train[[3]]).ravel()

knn = KNeighborsClassifier(metric='minkowski', n_neighbors=1, p=2)
knn.fit(trainX, trainY)

image = Image.open("d:/Programming/MachineLearning/photo/3.jpg")


draw = ImageDraw.Draw(image)
width = image.size[0] 
height = image.size[1]	
pix = image.load() 
for i in range(width):
        for j in range(height):
	    #print knn.predict( [ [ pix[i,j][0], pix[i,j][1], pix[i,j][2] ] ] )
	    if knn.predict( [ [ pix[i,j][0], pix[i,j][1], pix[i,j][2] ] ] ) == [1]:
	            draw.point( (i, j), (0, 255, 0) )

image.save("d:/Programming/MachineLearning/photo/res2.jpg", "JPEG")
del draw


#test = read_csv('d:/Downloads/test.csv')
#testX = np.asarray(test[range(1, 4)])
#testY = knn.predict(X_test)

#output = open('d:/Downloads/answer.csv',"w")
#print >> output, "id,y"
#for i, y in enumerate(testY):
#    print >> output, str(i + 1) + "," + str(y)

#output.close()