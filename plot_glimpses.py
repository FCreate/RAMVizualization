import pickle
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt

from utils import denormalize, bounding_box


def parse_arguments():
    arg = argparse.ArgumentParser()
    arg.add_argument("--plot_dir", type=str, required=True,
                     help="path to directory containing pickle dumps")
    arg.add_argument("--epoch", type=int, required=True,
                     help="epoch of desired plot")
    args = vars(arg.parse_args())
    return args['plot_dir'], args['epoch']


def main(plot_dir, epoch):

    # read in pickle files
    glimpses = pickle.load(
        open(plot_dir + "g_{}.p".format(epoch), "rb")
    )
    locations = pickle.load(
        open(plot_dir + "l_{}.p".format(epoch), "rb")
    )

    glimpses = np.concatenate(glimpses)

    # grab useful params
    print(plot_dir.split('_'))
    size = int(plot_dir.split('_')[2].split('x')[0])
    print(size)
    locations = np.array(locations)
    #locations = locations[:, :4, :]
    #glimpses = glimpses[:4, :,:]
    
    print(locations.shape)
    print(glimpses.shape)

    num_anims = len(locations)
    num_cols = glimpses.shape[0]
    print(num_cols)
    img_shape = glimpses.shape[1]
    
    # denormalize coordinates
    print((np.array(locations)).shape)
    print(glimpses.shape)

    coords = [denormalize(img_shape, l) for l in locations]
   
    fig, axs = plt.subplots(nrows=int(sqrt(num_cols)), ncols=int(sqrt(num_cols)),constrained_layout=True)
    fig.set_dpi(400)
    # plot base image
    for j, ax in enumerate(axs.flat):
        ax.imshow(glimpses[j], cmap="Greys_r")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    fig.set_constrained_layout_pads(w_pad=1./72., h_pad=1./72., hspace=0., wspace=0.)
    #for j, ax in enumerate(axs.flat):
        
    def updateData(i):
        color = 'r'
        print(i)
        co = coords[i]
        for j, ax in enumerate(axs.flat):
            for p in ax.patches:
                p.remove()
            c = co[j]
            rect = bounding_box(
                c[0], c[1], size, color
            )
            ax.add_patch(rect)

    # animate
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps = 0.5, metadata=dict(artist='Me'))
    anim = animation.FuncAnimation(fig, updateData, frames=num_anims, blit = False, interval=1)
    plt.show()
    # save as mp4
    name = plot_dir + 'epoch_{}.gif'.format(epoch)
    print (name)
    anim.save('epoch_{}.mp4'.format(epoch), writer=writer)
    #anim.save(name, writer='ffmpeg', codec='rawvideo')
    #anim.save(name, extra_args=['-vcodec', 'h264', '-pix_fmt', 'yuv420p'])


if __name__ == "__main__":
    args = parse_arguments()
    main(*args)
