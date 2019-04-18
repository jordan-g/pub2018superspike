import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

Ws = np.genfromtxt("Ws.txt", delimiter=" ")
Ys = np.genfromtxt("Ys.txt", delimiter=" ")
betas = np.genfromtxt("betas.txt", delimiter=" ")

if len(Ws.shape) == 1:
    Ws = Ws[np.newaxis, :]
    Ys = Ys[np.newaxis, :]
    betas = betas[np.newaxis, :]

print(Ws.shape)

# print(np.sum(betas[-1] == 0))

fig = plt.figure()
ax = plt.axes(xlim=(np.amin(Ws), np.amax(Ws)), ylim=(np.amin(betas), np.amax(betas)))
points = ax.scatter([], [], c='b', alpha=0.05)
plt.axvline(0, c='k')
plt.axhline(0, c='k')
plt.xlabel("w")
plt.ylabel("beta")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18)
plt.tight_layout()

def init():
    points.set_offsets([[], []])
    return points,

def animate(i):
    print(i)
    points.set_offsets(np.hstack([Ws[i][:, np.newaxis], betas[i][:, np.newaxis]]))
    return points,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Ws.shape[0], interval=200, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('weight_beta_animation.mp4', fps=3, extra_args=['-vcodec', 'libx264'])

fig = plt.figure()
ax = plt.axes(xlim=(np.amin(Ws), np.amax(Ws)), ylim=(np.amin(Ys), np.amax(Ys)))
points = ax.scatter([], [], c='b', alpha=0.05)
plt.axvline(0, c='k')
plt.axhline(0, c='k')
plt.xlabel("w")
plt.ylabel("y")
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.label.set_size(18)
ax.xaxis.label.set_size(18)
plt.tight_layout()

def init():
    points.set_offsets([[], []])
    return points,

def animate(i):
    print(i)
    points.set_offsets(np.hstack([Ws[i][:, np.newaxis], Ys[i][:, np.newaxis]]))
    return points,

anim = animation.FuncAnimation(fig, animate, init_func=init, frames=Ws.shape[0], interval=200, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('weight_beta_animation_2.mp4', fps=3, extra_args=['-vcodec', 'libx264'])