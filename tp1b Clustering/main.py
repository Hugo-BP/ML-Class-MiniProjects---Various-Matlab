import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statistics as st

# load data
N_DATASETS = 4

datasets = []
nclusters = []
labels = []

dataset_names = ['Aggregation', 'Compound', 'R15', 'Spiral']
distance_types = ['euclidean', 'manhattan', 'chebychev']


"""
Ex 10
Use the DBSCAN method from the sklearn.cluster module to cluster de data. Try different values of eps and min_samples . Visualize the results making use of
the function developed plot_clusters developed in Ex. 6. Discuss the following points:
1. In what way does the eps value influence the result of the algorithm? Does the same value work well for all datasets or it should be tuned for each dataset?
2. In what way does the min_samples value influence the result of the algorithm? Does the same value work well for all datasets or it should be tuned for each dataset?
3. Comment the differences in the results obtained by DBSCAN and kmeans .

ANSWER:
    1 - EPS - The maximum distance between two samples for one to be considered as in the neighborhood of the other. 
        This is not a maximum bound on the distances of points within a cluster. This is the most important DBSCAN parameter 
        to choose appropriately for your data set and distance function.
        EPS specifies how close points should be to each other to be considered a part of a cluster. It means that if the distance between
        two points is lower or equal to this value (eps), these points are considered neighbors.
    
    2 - MIN_SAMPLES - The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
        MIN_SAMPLES specifies the minimum number of points to form a dense region. For example, if we set the minPoints parameter as 5,
        then we need at least 5 points to form a dense region.
    
    1.5 / 2.5 - Parameter estimation - according to an article on DBSCAN Parameter Estimation Using Python, (https://medium.com/@tarammullin/dbscan-parameter-estimation-ff8330e3a3bd)
                we should set min samples such as:
                    The larger the data set, the larger the value of MinPts should be
                    If the data set is noisier, choose a larger value of MinPts
                    Generally, MinPts should be greater than or equal to the dimensionality of the data set
                    For 2-dimensional data, use DBSCAN’s default value of MinPts = 4
                and for EPS:
                    the best value would be the point of maximum curvature of a graph of the average 
                    distance between each point and its k nearest neighbors, where k = MIN_SAMPLES
                    
                We can confirm these two theories by first trying to use the same values for all the datasets and them looking at individual datasets, while comparing the results to the original plot:
                    A small EPS and high min_sample value benefits R15 greatly, ex eps = 0.4  min_sample = 15. 
                    Spiral on the other hand has the best results with a higher eps and min_sample value, ex eps = 3, min_sample = 6
                    A value of eps = 1.9 and min_samples = 13 gives great results on the first dataset, on the other hand.
                    With a value of eps = 0.52 and min_samples = 4 we can even find the cluster thats totally surrounded in the second dataset.
                We come to the conclusion that the value of these variables needs to be dependent on the data and its characteristics, like shape and noise.
                
    3 - kmeans uses a random factor, the centroids, so the results couldn't be as accurate in labeling the shapes like the original plot. 
        kmeans works with a radius with a centroid at its center, and so the shapes are always sphere-like. its an interesting 
        technique to find groups which have not been explicitly labeled in the data. For Kmeans clustering to work well, the following assumptions have to hold true: :
            the variance of the distribution of each attribute (variable) is spherical
            all variables have the same variance
            the prior probability for all k clusters are the same, i.e. each cluster has roughly equal number of observations
        
        DBSCAN does not require one to specify the number of clusters in the data a priori, as opposed to k-means.
        DBSCAN can find arbitrarily shaped clusters, and clusters inside of clusters, as shown with the second dataset
        DBSCAN has a notion of noise, and is robust to outliers.
        DBSCAN requires just two parameters which can be easily tuned.
        but 
        The quality of DBSCAN depends on the distance measure used. 
        DBSCAN cannot cluster data sets well with large differences in densities, as shown with the second dataset
             
    
"""


def ex10():
    from sklearn.cluster import DBSCAN

    fig, axes = plt.subplots(2, 2, figsize=[10, 10])
    fig.suptitle('sklearn.cluster DBSCAN')
    axs = axes.flatten()

    eps = 0.51
    min_samples = 4
    for ds in range(len(datasets)):
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(datasets[ds])
        axs[ds].scatter(datasets[ds].iloc[:, 0], datasets[ds].iloc[:, 1], c=clustering.labels_)

    plt.savefig('exercise_10_sklearn_cluster_DBSCAN_plot.png')
    plt.show()
    return


"""
Ex. 9
Run the kmeans function 10 times with each distance type (in each dataset). Compute report the average and standard deviation of the number of iterations taken by the
algorithm. Comment your results.

ANSWER:
Looking at the average of the values ​​of mean and standard deviation values ​​of the 3 datasets, it can be seen that, although the value of the standard deviation is lightly higher compared to the other two (so there is a lower density in the clusters), the mean is lower (so there is lower distances between the centroids and the data).
Therefore, when evaluating all the datasets, Euclidean k-means is preferred.
Looking at dataset 0: the Manhattan k-means represents the best compromise between mean and standard deviation;
Looking at dataset 1: the Manhattan k-means is the best for both the mean and the standard deviation;
Looking at dataset 2: the Euclidean k-means represents the best compromise between mean and standard deviation;
Looking at dataset 3: the Euclidean k-means is the best for both the mean and the standard deviation.

"""


def ex9():
    for dt in ['euclidean', 'manhattan', 'chebychev']:
        print(dt)
        it_arr_tot = []
        for ds in range(len(datasets)):
            it_arr = []
            for i in range(10):
                # calc kmeans
                lab, centroids, it = kmeans(datasets[ds], nclusters[ds], dt)
                # add iterations to array
                it_arr.append(it)
                it_arr_tot.append(it)
            print('dataset ', ds, ':')
            print('mean = ', st.mean(it_arr), '\nstdev = ', st.stdev(it_arr))
        print('dataset average:')
        print('mean = ', st.mean(it_arr_tot), '\nstdev = ', st.stdev(it_arr_tot))
        print('\n')
    # return average and standard deviation
    return


"""
Ex. 8
The code bellow generates 3 figures with with the final result of applying the kmeans function to the data using the three different types of distance ( euclidean ,
manhattan and chebychev ). Did the distance type influence the results?

ANSWER:
Results that are obtained after the implementation of K-means using 3 various distance metrics are shown in the plots.
By looking at the plots, we can see how the position of the centroids is different according to the type of distance considered

"""


def ex8():
    for dt in ['euclidean', 'manhattan', 'chebychev']:
        fig, axes = plt.subplots(2, 2, figsize=[10, 10])
        fig.suptitle('Distance type: %s' % dt)
        axs = axes.flatten()
        for ds in range(len(datasets)):
            lab, centroids, it = kmeans(datasets[ds], nclusters[ds], dt)

            axs[ds].scatter(datasets[ds].iloc[:, 0], datasets[ds].iloc[:, 1], c=lab)
            axs[ds].scatter([centroids[:, 0]], [centroids[:, 1]], marker='x', color='r')

    plt.savefig('exercise_8_cluster_plot.png')
    plt.show()
    return


"""
Ex. 7
Create a function kmeans( data, n, dist_type='euclidean' ) that, given a dataset with shape [n_samples, d] , 
clusters the data into n sets using the kmeans algorithm. Make use of the functions you developed previously; 
you cannot resort to sklearn nor other implementations. The function should return the final
labels, the final centroid coordinates and the number of iterations run.
"""


def kmeans(data, k, dist_type='euclidean', max_iter=20):
    old_labels = np.empty((data.shape[0], 1))
    new_labels = np.random.randint(k, size=(data.shape[0], 1))
    # initialize randomly the centroids
    min_coords = [0, 0]
    max_coords = [25, 25]
    centroids = gen_random_centroids(k, min_coords, max_coords)

    i = 0
    # REPEAT UNTIL i REACHES MAX_ITERATIONS OR UNTIL NEW LABELS AND OLD LABELS ARE EQUAL
    # loop at most max_iter
    while i < max_iter:
        if (new_labels == old_labels).all():
            break
        old_labels = new_labels
        # get new_labels from data using the current centroid locations
        new_labels = label_data(data, centroids, dist_type)
        # update the centroid locations based on the new labels
        centroids = update_centroids(data, new_labels, centroids)
        # increment loop
        i += 1
        # if updated new_labels are equal to old labels then break loop early


    # return updated labels & centroids + amount of iterations
    return new_labels, centroids, i


""""
Ex. 6
Create a function plot_clusters( data, labels, centroids=None ) that plots a scatter plot of the data, coloring the points of each cluster with a different color,
and marking the centroids, if provided.
"""


def plot_clusters(data, labels, centroids=None):
    """
    COPYPASTE FROM EX.5
    """
    # turn into np.array so its easier to handle
    data = np.asarray(data)
    # get all unique label types inside the array, ex. [1,0,1,0,1,0,1,2,2,2] --> [0,1,2]
    unique_labels = np.unique(labels)
    # for each unique label, get the indexes where that label is present inside the labels array.
    for label in unique_labels:
        # random color for the plot
        color = np.random.rand(3, )
        # these will be the indexes of each cluster
        cluster_indexes = np.where(labels == label)
        # this is where the cluster coordinates will be stored temporarily
        cluster = []
        # for each INDEX inside the cluster_indexes array, get the respective vector of coordinates at data[INDEX]
        for i in cluster_indexes[0]:
            # save these vectors in the cluster array
            cluster.append(data[i])
        # turn cluster into np.array so its easier to handle
        cluster = np.asarray(cluster)
        """
        COPYPASTE FROM EX.5
        """
        # scatterplot the cluster
        plt.scatter(cluster[:, 0], cluster[:, 1], color=color)

        if centroids is not None:
            centroids = np.asarray(centroids)
            # TODO is this [label-1] OR [label], if not using the global labels change to [label].
            plt.scatter(centroids[label:, 0], centroids[label:, 1], marker='^', color=color, s=80)

    plt.title('Exercise 6: Plot expected clusters with||without Centroids')
    plt.savefig('exercise_6_cluster_plot.png')
    plt.show()
    return


"""
Ex. 5
Create a function update_centroids( data, labels, centroids ) that updates the centroids coordinates based on the mass center of the data records
associated with him. If a centroid has no record associated to him, its value must remain unchanged.
"""


def update_centroids(data, labels, centroids):
    # turn data into np.array so its easier to handle
    data = np.asarray(data)
    # get all unique label types inside the array, ex. [1,0,1,0,1,0,1,2,2,2] --> [0,1,2]
    unique_labels = np.unique(labels)

    # for each unique label, get the indexes where that label is present inside the labels array.
    for label in unique_labels:
        # these will be the indexes of each cluster
        cluster_indexes = np.where(labels == label)
        # this is where the cluster coordinates will be stored temporarily
        cluster = []
        # for each INDEX inside the cluster_indexes array, get the respective vector of coordinates at data[INDEX]
        for i in cluster_indexes[0]:
            # save these vectors in the cluster array
            cluster.append(data[i])

        # turn cluster into np.array so its easier to handle
        cluster = np.asarray(cluster)
        # to calculate the centroid of an array of 2D coordinates you can just calculate the mean
        new_centroid = cluster.mean(axis=0)
        # update the centroids array
        # TODO does this always receives labels from 0 to n-1 ?
        centroids[label] = new_centroid

    # return updated array
    return centroids


"""
Ex. 4
Create a function label_data( data, centroids, dist_type='euclidean' ) that attributes a cluster label for each data record based on its distance to the
centroids. The data shape is [n_samples, d] and the centroids shape is [n, d] . You should output an array of shape [n_samples, 1] with values ranging
between 0 and n-1 , corresponding to the index of the closest centroid for each data record. Please consider different types of distance metrics, making use of the
function developed in Ex. 2
"""


def label_data(data, centroids, dist_type='euclidean'):
    expected_labels_array = []
    # turn data into np.array so its easier to handle
    data = np.asarray(data)
    centroids = np.asarray(centroids)
    # loop through all coords in data
    for coord in data:
        # this is where the distances between current coord and ALL centroids are saved. It is cleared every new coord
        distances = []
        # for a certain coord, loop through all centroids
        for centroid in centroids:
            # add all possible distances to the distances array. use calc_dist() to calc said distance
            distances.append(calc_dist(coord, centroid, dist_type))

        # get the index of the lowest distance value for the current coord
        expected_label = distances.index(min(distances))
        # save expected label. index matters.
        expected_labels_array.append([expected_label])

    # output the labels as an array
    return np.asarray(expected_labels_array)


"""
Ex. 3
Create a function gen_random_centroids( n, min_coords, max_coors ) that generates n random points within the coordinate limits provided in the arrays
min_coords and max_coords (both of shape = [1,d] , with d being the number of coordinates of each point - in our datasets, d=2 ). The function must output a
matrix of shape [n, d] , where each row contains the coordinates of a centroid.

# example
print(gen_random_centroids(3, [2, -3], [12, 5]))
"""


def gen_random_centroids(n, min_coords, max_coords):
    centroid_coord_array = []
    # turn vectors into np array
    min_coords = np.asarray(min_coords)
    max_coords = np.asarray(max_coords)

    # create n random coords between values of min and max coords, save them on array
    for i in range(n):
        coord = min_coords + np.random.uniform(0, 1) * (max_coords - min_coords)
        centroid_coord_array.append(coord)

    # output array as a matrix
    return np.asmatrix(centroid_coord_array)


"""
Ex. 2
Create a function calc_dist( xi, xj, dist_type ) that, given two feature vectors ( xi and xj ) and the type of distance metric to use ( dist_type , with
possible values: euclidean, manhattan, chebychev ), computes and returns the correspondent distance value between the two feature vectors. NOTE: You may not
resort to third-party predefined distance functions, such as the ones provided by the scipy.spatial.distance module. You must compute the distance from the data,
using only simple mathematical and algebric functions, such as sum, sqrt, abs and so on.
"""


# xi, xj each are a row of ex. aggregation (x,y) no label
def calc_dist(xi, xj, dist_type):
    if dist_type == 'euclidean':
        # euclidean formula
        euc = np.sqrt(np.sum((xi - xj) ** 2))
        return euc

    elif dist_type == 'manhattan':
        # manhattan formula
        man = np.sum(np.abs(xi - xj))
        return man

    else:  # chebychev
        # chebychev formula
        ch = np.max(np.abs(xi - xj))
        return ch


"""
Ex. 1
Load the datasets into memory and extract, for each dataset, the number of true clusters in the data. You should populate de datasets and nclusters arrays so they
end up with four cells, one for each dataset, with the dataframes ( shape=[nsamples, d=2] and the number of clusters, respectively. Then, create a figure with four
scatter plots showing the spatial distribution of the points in each dataset. You might find useful the following functions: iloc , from pandas ; subplots , scatter
and set_title from matplotlib's pyplot . The final result should be similar to the following image:
"""


def plot_clusters_init():
    fig, axes = plt.subplots(2, 2, figsize=[10, 10])
    # subplot (0,0)
    x = 0
    y = 0
    for name in dataset_names:
        # load data
        ds = pd.read_csv(name + '.txt', sep="\t", header=None)
        # name columns
        ds.columns = ["x", "y", "c"]

        # get unique clusters in dataset
        unique_shapes = ds["c"].unique()

        # for each unique cluster shape of the dataset:
        for shape in unique_shapes:
            # get the points that are labeled that shape
            cluster = ds[ds["c"] == shape]  # ex. in dataset 1, get all points labeled "1", aka that make up cluster 1

            # scatterplot the cluster in the respective subplot
            axes[x, y].scatter(cluster['x'], cluster['y'])
            # name subplot
            axes[x, y].set_title(name)

        # give each dataset a different subplot
        if x == 0 and y == 0:
            # subplot (0,1)
            y = 1
        elif x == 0 and y == 1:
            # subplot (1,0)
            y = 0
            x = 1
        elif x == 1 and y == 0:
            # subplot (1,1)
            y = 1
            x = 1

    plt.savefig('exercise_1_dataset_plot.png')
    plt.show()

    return


"""
MAIN
"""


def main():
    # read datasets.txt amd fill arrays according to PDF
    for name in dataset_names:
        # load data from txt
        ds = pd.read_csv(name + '.txt', sep="\t", header=None)
        # fill global data arrays
        globals()['datasets'].append(ds.iloc[:, :2])
        globals()['labels'].append(ds.iloc[:, -1])
        unique_shapes = (ds.iloc[:, 2]).unique()
        globals()['nclusters'].append(len(unique_shapes))

    # ex.1 requires a plot of the original datasets. This is also saved as a .png file
    plot_clusters_init()

    # exercise 8
    ex8()

    # exercise 9
    ex9()

    # exercise 10
    ex10()


    """
    
    
    THIS DEMO CODE IS JUST FOR TESTING PURPOSES:
    
    
    """

    """
    # DEMO for dist_calc(): calculates all 3 types of distances between row 3 and 32 of dataset no.3
    for dist in distance_types:
        dataset_target = 2  # 0 - 3
        row_i = 3  # 0 - dataset len -1
        row_j = 32  # 0 - dataset len -1

        # vector 1
        xi = np.asarray([datasets[dataset_target][0][row_i], datasets[dataset_target][1][row_i]])
        # vector 2
        xj = np.asarray([datasets[dataset_target][0][row_j], datasets[dataset_target][1][row_j]])
        # calculate distance
        # print(xi,xj)
        print(dist, row_i, row_j, calc_dist(xi, xj, dist))

    # DEMO for gen_random_centroids(): (n = number of coord output, vector min, vector max)
    centroids = gen_random_centroids(3, [2, -3], [12, 5])
    print('centroids test\n', centroids)

    # DEMO for label_data(): for real data use datasets[0 or 1 or 2 or 3] as ex_data, gen_random_centroids() output as ex_centroids, default distance used is euclidean.
    ex_data = pd.DataFrame(np.array([[0, 0], [1, 0], [2, 0], [1, 0], [1, 1], [1, 2]]))
    ex_centroids = np.array([[0.2, 0.2], [1.6, 0.6], [5.0, 5.0]])
    ex_labels = label_data(ex_data, ex_centroids)
    print('ex_labels test\n', ex_labels)

    # DEMO for update_centroids():
    ex_data = pd.DataFrame(np.array([[0, 0], [1, 0], [2, 0], [1, 0], [1, 1], [1, 2]]))
    ex_labels = np.array([[1], [1], [0], [1], [0], [0]])
    ex_centroids = np.array([[0.2, 0.2], [1.6, 0.6], [5.0, 5.0]])
    new_centroids = update_centroids(ex_data, ex_labels, ex_centroids)
    print('new_centroids test\n', new_centroids)

    # DEMO for plot_clusters() exercise 7: uses dataset 1 labels 1 and no centroids
    plot_clusters(ex_data, ex_labels, new_centroids)
    # TODO change update_centroids() label to label-1 to make this test work
    # plot_clusters(datasets[0], labels[0], update_centroids(datasets[0], labels[0],np.array([[6.4, 3.2], [17, 7], [7.5, 12.0], [33, 8], [33,25], [22, 22.0], [8.6, 22.6]])) )

    # DEMO for kmeans()
    # TODO change kmeans() label to label-1 to make this test work
    # print('kmeans test = ', kmeans(datasets[0], 5))

    l, c, i = kmeans(ex_data, 3)
    print('kmeans test:')
    print('labels:\n', l)
    print('clusters:\n', c)
    print('i = ', i)
    
    """


    return


# run main
main()
