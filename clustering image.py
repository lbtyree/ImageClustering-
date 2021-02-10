import matplotlib.pyplot as plt
from scipy import ndimage
import sklearn 
from sklearn import cluster

def create():   
    #initial read of image 
    first_img = plt.imread("trees.png")
    
    #create 2d imagee from input image 
    second_img = first_img.reshape(first_img.shape[0]*first_img.shape[1], first_img.shape[2])
   
    #choose 3 representative k values
    #using .Kmeans(3).fit to find the 3 best representative k vals 
    k=3
    km = cluster.KMeans(k).fit(second_img) 

    #show images side by side 
    img = plt.figure(figsize=(12, 7))
    one = img.add_subplot(2,2,1)
    one.imshow(first_img) #initial
    two = img.add_subplot(2,2,2)
    #km.cluster_centers_and km.labels_returns centers and labels of clusters found by .fit 
    two.imshow(km.cluster_centers_[km.labels_].reshape(first_img.shape[0], first_img.shape[1], first_img.shape[2])) #clustered 
    plt.show()

if __name__ == "__main__":
    create()
