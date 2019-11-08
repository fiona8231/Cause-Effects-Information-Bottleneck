import logging
from matplotlib.backends.backend_pdf import PdfPages
from pylab import *
import pickle
import pylab
logger = logging.getLogger("root")


def plot_information_curve_line(file, path, ixt, iyt):
    logger.info("Evaluate artificial information curve")

    # create pdf
    with PdfPages(path + "ic.pdf") as pdf:
        fig = plt.figure()

        # open file
        with open(file, 'r') as f:
            data = pickle.load(f)

        plt.plot(data[2], data[0], color='#FF9900',
                 label=r'Information Curve', linewidth=3, linestyle='-')

        # set legend

        plt.legend(loc="lower right")

        # set axis label
        plt.ylabel(r'$I(Z;(T,Y))$')
        plt.xlabel(r'$I(X;Z)$')

        ax = plt.gca()
        #ax.set_xlim([0, 6])
        #ax.set_ylim([0, 0.6])
        ax.set_axis_bgcolor('#f6f6f6')

        # set background
        ax = plt.gca()
        ax.set_axis_bgcolor('#f6f6f6')
        x = ixt
        y = iyt
        # colors = np.random.rand(N,N)
        area = 60

        plt.scatter(x, y, color='g', marker='.', s=area, alpha=0.5)
        pdf.savefig(fig)
        plt.show()
        plt.close()


def plot_information_curve_scatter(ixt, iyt):
    plt.gca()

    plt.figure(facecolor="white")
    x = ixt
    y = iyt
    area = 60
    plt.scatter(x, y, color='g', marker='.', s=area, alpha=0.5)

    # Axis label
    plt.ylabel(r'$I(Z;(T,Y))$')
    plt.xlabel(r'$I(X;Z)$')
    #plt.axis([-1, 8, -0.1, 0.6])
    plt.show()
