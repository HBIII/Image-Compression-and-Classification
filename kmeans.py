import numpy as np


def get_k_means_plus_plus_center_indices(n, n_cluster, x, generator=np.random):
    '''
    :param n: number of samples in the data
    :param n_cluster: the number of cluster centers required
    :param x: data-  numpy array of points
    :param generator: random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results

    :return: the center points array of length n_clusters with each entry being index to a sample
             which is chosen as centroid.
    '''

    centers = []
    centers.append(generator.randint(0, n))

    for i in range(1, n_cluster):

        centroids = x[centers]
        distances = np.linalg.norm(x - centroids[:, None], axis=2)
        farthest_x_index = np.argmax(np.min(distances, axis=0))
        centers.append(farthest_x_index)

    print("[+] returning center for [{}, {}] points: {}".format(n, len(x), centers))
    return centers

def get_lloyd_k_means(n, n_cluster, x, generator):
    return generator.choice(n, size=n_cluster)

class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results
    '''
    def __init__(self, n_cluster, max_iter=100, e=0.0001, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator

    def fit(self, x, centroid_func=get_lloyd_k_means):

        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a length (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates a Int)
        '''
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        
        N, D = x.shape

        self.centers = centroid_func(len(x), self.n_cluster, x, self.generator)

        max_val = 10000000

        centroids = x[self.centers]
        J = max_val
        y = np.zeros(N)

        for num_of_updates in range(1, self.max_iter):
            
            y = np.argmin(np.linalg.norm(x-centroids[:, None], axis=2), axis=0)
            J_new = 0
            
            for j in range(self.n_cluster):
                J_new += np.sum(np.linalg.norm(centroids[j]-x, axis=1)*(y == j))

            if abs(J_new-J) <= self.e:
                return centroids, y, num_of_updates

            J = J_new
            
            for cluster in range(self.n_cluster):
                cluster_indices = (y==cluster)
                centroids[cluster] = np.sum(x[cluster_indices], axis=0) / np.sum(cluster_indices==True)

        return centroids, y, self.max_iter


class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int)
            e - error tolerance (Float)
            generator - random number generator from 0 to n for choosing the first cluster at random
            The default is np.random here but in grading, to calculate deterministic results
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6, generator=np.random):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e
        self.generator = generator


    def fit(self, x, y, centroid_func=get_lloyd_k_means):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
                centroid_func - To specify which algorithm we are using to compute the centers(Lloyd(regular) or Kmeans++)

            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by
                    majority voting (N,) numpy array)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        self.generator.seed(42)
        N, D = x.shape

        k_means = KMeans(n_cluster=self.n_cluster, max_iter=self.max_iter, e=self.e)

        centroids, labels, _ = k_means.fit(x, centroid_func)

        centroid_labels = np.zeros(self.n_cluster)
        for i in range(self.n_cluster):
            centroid_labels[i] = np.argmax(np.bincount(y[labels==i]))

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (
            self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(self.n_cluster)

        assert self.centroids.shape == (
            self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function
            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        self.generator.seed(42)
        N, D = x.shape

        labels_indices = np.argmin(np.linalg.norm(x[:, None]-self.centroids, axis=2), axis=1)
        labels = self.centroid_labels[labels_indices]

        return np.array(labels)
        

def transform_image(image, code_vectors):
    '''
        Quantize image using the code_vectors

        Return new image from the image by replacing each RGB value in image with nearest code vectors (nearest in euclidean distance sense)

        returns:
            numpy array of shape image.shape
    '''

    assert image.shape[2] == 3 and len(image.shape) == 3, \
        'Image should be a 3-D array with size (?,?,3)'

    assert code_vectors.shape[1] == 3 and len(code_vectors.shape) == 2, \
        'code_vectors should be a 2-D array with size (?,3)'

    new_im = np.zeros(np.shape(image))

    for i in range(np.shape(image)[0]):
        for j in range(np.shape(image)[1]):
            new_im[i,j,:] = code_vectors[np.argmin(np.linalg.norm(image[i,j,:]-code_vectors, axis=1))]

    return new_im

