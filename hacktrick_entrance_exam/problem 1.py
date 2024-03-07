from matplotlib.image import imread
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def segment_image():
    image = imread("ireland.jpg")
    image = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2).fit(image)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    seg_img = centers[labels].reshape(640, 550, 3)
    seg_img = seg_img.astype(int)
    return {'segmented_image': seg_img,'clusterer': kmeans}


img = segment_image()