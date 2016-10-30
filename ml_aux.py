"""Auxiliar module for the ML class."""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from keras.datasets import cifar10, mnist

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.datasets.samples_generator import make_blobs


def show_ml_examples():
    """Plot some Machine Learning toy examples."""
    fig = plt.figure(figsize=(15, 10))

    ax = fig.add_subplot(2, 3, 1)
    ax.text(0, 0.45, 'Supervised', fontsize=30)
    ax.axis('off')

    plt.subplot(2, 3, 2)
    plt.title('Classification')
    _plot_classification_example()

    plt.subplot(2, 3, 3)
    plt.title('Regression')
    _plot_regression_example()

    ax = fig.add_subplot(2, 3, 4)
    ax.text(0, 0.45, 'Unsupervised', fontsize=30)
    ax.axis('off')

    plt.subplot(2, 3, 5)
    plt.title('Clustering')
    _plot_clustering_example()

    plt.show()


def _plot_regression_example():
    """Plot a regression example."""
    X = np.arange(0, 10, 0.5)
    y = 1.5 * X + 2 + np.random.randn(len(X))
    X = X[:, np.newaxis]

    reg = LinearRegression(fit_intercept=True)
    reg.fit(X, y)

    plt.plot(X, y, 'o')
    plt.plot([0, 10], [reg.intercept_, 10 * reg.coef_[0] + reg.intercept_])
    plt.xlabel('Area of the house')
    plt.ylabel('Price')


def _plot_classification_example():
    """Plot the classification example."""
    X, y = make_blobs(n_samples=200, centers=2, random_state=0,
                      cluster_std=0.60)

    clf = SVC(kernel='linear')
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='spring')
    plot_svc_decision_function(clf)
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=200,
                facecolors='none')
    plt.ylabel('Glucose levels')
    plt.xlabel('Blood pressure')


def _plot_clustering_example():
    """Plot the clustering example."""
    X, y = make_blobs(n_samples=200, centers=3, random_state=0,
                      cluster_std=0.60)
    X -= X.min(0)

    clu = KMeans(n_clusters=3)
    y_kmeans = clu.fit_predict(X)

    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='spring')
    plt.scatter(clu.cluster_centers_[:, 0], clu.cluster_centers_[:, 1], s=100)
    plt.xlabel('Visits to the store/year')
    plt.ylabel('Number of purchases/year')


def plot_svc_decision_function(clf, ax=None):
    """Plot the decision function for a 2D SVC."""
    plot_decision_function(clf.decision_function, [-1, 0, 1], ax)


def plot_proba_function(clf, ax=None):
    """Plot the decision function for a classifier with predict_proba."""
    fn = lambda x: clf.predict_proba(x)[0][0]
    plot_decision_function(fn, [0, 0.5, 1], ax)


def plot_decision_function(fn, levels, ax=None):
    """Plot the decision function for a given classifier function."""
    if ax is None:
        ax = plt.gca()
    x = np.linspace(plt.xlim()[0], plt.xlim()[1], 30)
    y = np.linspace(plt.ylim()[0], plt.ylim()[1], 30)
    Y, X = np.meshgrid(y, x)
    P = np.zeros_like(X)
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            data_point = np.array([xi, yj]).reshape(1, -1)
            P[i, j] = fn(data_point)
    # plot the margins
    ax.contour(X, Y, P, colors='k',
               levels=levels, alpha=0.5,
               linestyles=['--', '-', '--'])


def plot_logistic_regression(N=5, C=1, fit_intercept=True, penalty='l2'):
    """Plot Logistic Regression and its decision function.

    Parameters:
    N - Number of datapoints used to train the SVM.
    C - the regularization term.
    """
    X, y = make_blobs(n_samples=200, centers=2, random_state=0,
                      cluster_std=0.80)

    X_train, y_train = X[:N], y[:N]
    X_test, y_test = X[N:], y[N:]

    clf = LogisticRegression(C=C, fit_intercept=fit_intercept, penalty=penalty)
    clf.fit(X_train, y_train)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='spring')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, s=50, cmap='spring',
                alpha=0.2)
    plt.xlim(-1, 4)
    plt.ylim(-1, 6)
    plot_proba_function(clf, plt.gca())

    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test) if len(X_test) > 0 else 'NA'
    plt.title('Train Accuracy = {0}; Test Accuracy = {1}; coef = {2}'.format(
        train_score, test_score, clf.coef_))


def train_logistic_regression(X, y):
    """Train a logistic regression classifier."""
    scaler = StandardScaler()
    clf = SGDClassifier(loss='log', n_iter=1, alpha=1, eta0=1e-6,
                        learning_rate='constant')
    pipeline = Pipeline([('scaler', scaler), ('clf', clf)])
    return pipeline.fit(X.astype(np.float64), y)


def plot_weights(pipeline, shape, classes, size=(9, 12)):
    """
    Show the weights of a model.

    Parameters:
    - pipeline: the model.
    - shape: the shape of the images.
    - classes: the classes to show.
    - size: the size of the figure.
    """
    plt.figure(figsize=size)
    for i, c in enumerate(classes):
        clf = pipeline.named_steps['clf']
        coef = clf.coef_[i]
        coef -= coef.min()
        coef /= coef.max()
        plt.subplot(4, 3, i + 1)
        plt.title(c)
        plt.imshow(coef.reshape(shape))
        plt.axis('off')

    plt.show()


def show_dataset(X, y, shape, classes, size=(9, 12)):
    """
    Show the given dataset.

    Parameters:
    - X: the data.
    - y: the labels.
    - shape: the shape of the images.
    - classes: the different classes in y to show.
    - size: the size of the figure.
    """
    plt.figure(figsize=(9, 12))
    for i, c in enumerate(classes):
        img = X[y == i][0]
        plt.subplot(4, 3, i+1)
        plt.title(c)
        plt.imshow(img.reshape(shape))
        plt.axis('off')
    plt.show()


def neural_network_exercise():
    """Train a neural network on the MNIST database."""
    (X_train, y_train), (X_test, y_test) = load_MNIST()

    clf = MLPClassifier([150])
    clf.fit(X_train, y_train)

    print 'Accuracy on train set = {0}'.format(clf.score(X_train, y_train))
    print 'Accuracy on test set = {0}'.format(clf.score(X_test, y_test))

    return clf


def load_MNIST():
    """Load the MNIST dataset."""
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape((len(X_train), 28*28))
    X_test = X_test.reshape((len(X_test), 28*28))

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return (X_train, y_train), (X_test, y_test)


def load_CIFAR10():
    """Load the CIFAR-10 dataset."""
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape((len(X_train), 3*32*32))
    X_test = X_test.reshape((len(X_test), 3*32*32))
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    return (X_train, y_train), (X_test, y_test)


def plot_confusion_matrix(confusion_matrix):
    """Plot the given confusion_matrix."""
    ax = sns.heatmap(confusion_matrix, annot=False)
    ax.set(xlabel="predicted", ylabel="true")


def plot_wrong_predictions(clf, X, y, width=2, height=2, shape=(28, 28)):
    """Plot the wrongly classified digits with the wrong and true value."""
    y_pred = clf.predict(X)

    idx = y != y_pred
    X_wrong = X[idx]
    y_pred_wrong = y_pred[idx]
    y_true_wrong = y[idx]

    n = width * height
    if n > len(X_wrong):
        n = len(X_wrong)

    for i in range(n):
        ax = plt.subplot(height, width, i+1)
        plt.imshow(X_wrong[i].reshape(shape))
        ax.text(0.95, 0.01, '{0}'.format(y_pred_wrong[i]),
                verticalalignment='bottom', horizontalalignment='right',
                color='red', fontsize=15, transform=ax.transAxes)
        ax.text(0.01, 0.95, '{0}'.format(y_true_wrong[i]),
                color='green', fontsize=15, transform=ax.transAxes)
        plt.axis('off')

    plt.show()
