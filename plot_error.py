"""
Plot test error for SGD and L-BFGS on two data sets.
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rc
rc('font', **{'family':'sans-serif', 'sans-serif':['Arial'], 'size':14})
#rc('text', usetex=True)
rc('xtick', **{'labelsize':12})
rc('ytick', **{'labelsize':12})


#-----------------------------------------------------------------------------
# PRE-PROCESS DATA
#-----------------------------------------------------------------------------
# (USPS-N, Web)
sgd_error = (5.32, 8.75)
sgd_std = (3.23, 4.3)
lfbgs_error = (3.21, 5.45)
lfbgs_std = (0.9, 1.4)

#
ind = np.arange(2)
width = 0.25

#-----------------------------------------------------------------------------
# PLOT DATA
#-----------------------------------------------------------------------------
plt.figure()
ax = plt.subplot(111)

rects1 = ax.bar(ind, sgd_error, width, yerr=sgd_std, 
                facecolor='#999999', ecolor='black', hatch="")
rects2 = ax.bar(ind + width, lfbgs_error, width, yerr=lfbgs_std,
                facecolor='#eeeeee', ecolor='black', hatch="//")

# frame
ax.set_ylim([0, 14])

# labels
ax.set_xlabel('Data Set')
ax.set_ylabel('Test Error [%]')
ax.set_xticks(ind + width)
ax.set_xticklabels(('USPS-N', 'Web'))

# legend
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend((rects1[0], rects2[0]), ('SGD', 'L-BFGS'), loc="center left", bbox_to_anchor=(1.0, 0.5))

plt.savefig("test_error.pdf")
plt.show()
