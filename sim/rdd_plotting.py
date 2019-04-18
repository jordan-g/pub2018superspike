import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

thr = -50e-3

rdd_param_0 = np.loadtxt("rdd_param_0.txt", delimiter="\n")
rdd_param_1 = np.loadtxt("rdd_param_1.txt", delimiter="\n")
rdd_param_2 = np.loadtxt("rdd_param_2.txt", delimiter="\n")
# print(rdd_param_2)
rdd_param_3 = np.loadtxt("rdd_param_3.txt", delimiter="\n")
rdd_feedback = np.loadtxt("rdd_feedback.txt", delimiter="\n")
max_drive = np.loadtxt("max_drive.txt", delimiter="\n")
betas = np.loadtxt("beta.txt", delimiter="\n")
y = np.loadtxt("y.txt", delimiter="\n")
Ws = np.genfromtxt("Ws.txt", delimiter=" ")

below_indices = [ a for a in range(len(max_drive)) if max_drive[a] < thr]
above_indices = [ a for a in range(len(max_drive)) if max_drive[a] >= thr]

fig = plt.figure()
ax = plt.axes(xlim=(np.amin(max_drive)-thr, np.amax(max_drive)-thr), ylim=(-3, 3))
above_points = ax.scatter([], [], c='b', alpha=0.2)
below_points = ax.scatter([], [], c='r', alpha=0.2)
below_mean_line, = ax.plot([], [], 'r')
above_mean_line, = ax.plot([], [], 'b')
plt.axvline(0, c='k')
plt.xlabel(r'$u_{\mathrm{max}}^{(1)}$')
plt.ylabel(r'$J_{\mathrm{sum}}^{(1)}$')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
title = ax.text(.5, 1.05, r'$\beta=0$', transform = ax.transAxes, va='center', ha='center', fontsize=18)
ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18)
plotted_above_indices = []
plotted_below_indices = []
plt.tight_layout()
# plt.show()

def init():
    below_points.set_offsets([[], []])
    above_points.set_offsets([[], []])
    below_mean_line.set_data([], [])
    above_mean_line.set_data([], [])
    title.set_text("beta = 0")
    return below_points, above_points, below_mean_line, above_mean_line, title

def animate(i):
    # print(i)
    if i > 0:
        i = i-1
        
        if i in above_indices:
            plotted_above_indices.append(i)
        else:
            plotted_below_indices.append(i)

        # print((np.hstack([np.array(max_drive)[plotted_above_indices][:, np.newaxis], np.array(rdd_feedback)[plotted_above_indices][:, np.newaxis]])).shape)
        # print((np.hstack([np.array(max_drive)[plotted_below_indices][:, np.newaxis], np.array(rdd_feedback)[plotted_below_indices][:, np.newaxis]])).shape)
        above_points.set_offsets(np.hstack([np.array(max_drive)[plotted_above_indices][:, np.newaxis] - thr, np.array(rdd_feedback)[plotted_above_indices][:, np.newaxis]]))
        below_points.set_offsets(np.hstack([np.array(max_drive)[plotted_below_indices][:, np.newaxis] - thr, np.array(rdd_feedback)[plotted_below_indices][:, np.newaxis]]))

        x = np.linspace(np.amin(max_drive) - thr, 0, 100)
        y_below = rdd_param_3[i]*x + rdd_param_1[i]
        # print((y_below))
        below_mean_line.set_data(x, y_below)
        x = np.linspace(0, np.amax(max_drive) - thr, 100)
        y_above = rdd_param_2[i]*x + rdd_param_0[i]
        above_mean_line.set_data(x, y_above)
        beta = rdd_param_0[i] - (rdd_param_1[i])
        # print(beta)
        title.set_text("beta = {:.3f} | {} | {} | {}".format(beta, betas[i], y[i], Ws[0, 1]))
    else:
        below_points.set_offsets(np.zeros((1, 2))*np.nan)
        above_points.set_offsets(np.zeros((1, 2))*np.nan)
        x = np.linspace(np.amin(max_drive), thr, 100)
        y_below = np.ones(x.shape)
        below_mean_line.set_data(x, y_below)
        x = np.linspace(thr, np.amax(max_drive), 100)
        y_above = np.ones(x.shape)
        above_mean_line.set_data(x, y_above)
        title.set_text("beta = 0")
    return below_points, above_points, below_mean_line, above_mean_line, title

print(len(max_drive))
anim = animation.FuncAnimation(fig, animate, init_func=init, frames=min(1000, len(max_drive)), interval=200, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('RDD_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# plt.show()