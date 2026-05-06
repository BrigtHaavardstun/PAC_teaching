import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from dataclasses import dataclass
import math
display_w, display_h = 600, 600
w, h = 5000, 5000   # grid resolution — fewer cells = bigger "pixels"


@dataclass
class Circle:
    x: int
    y: int
    r: float


def cell_color(x, y):
    # Return an RGB tuple with values in [0, 1]
    return (x / w, y / h, 0.5)


def display_img(imgs, tag):

    n_img = len(imgs)
    fig, ax = plt.subplots(
        1,
        n_img,
        figsize=(n_img * display_w / 100 + 1.2, display_h / 100),
        dpi=100,
        constrained_layout=True
    )

    if n_img == 1:
        ax = [ax]

    for i, img in enumerate(imgs):
        ax[i].imshow(img, origin='lower')
        ax[i].axis('off')
        ax[i].set_box_aspect(1)

    sm = plt.cm.ScalarMappable(
        cmap='RdYlGn', norm=plt.Normalize(vmin=0, vmax=100))
    fig.colorbar(sm, ax=ax, location='right', fraction=0.04, pad=0.02)
    # cax.set_ylabel(
    #    'probability of learner believing c is consistent with c^*(x)=b', labelpad=10, wrap=True)

    # ax.set_title(
    #    "Probability of including left current cirlce (grey)\n given right target circle (blue) for each point")
    plt.savefig(f"img/{tag}.png")
    # plt.show()


def scalar_to_color(v: float) -> tuple:
    import matplotlib
    cmap = matplotlib.colormaps['RdYlGn']
    color = cmap(float(v))  # returns RGBA tuple for v in [0, 1]
    return color[:3]


def build_img_array(w: int, h: int, c_target: Circle, c_current: Circle, q: float, err: Callable[[int, int, Circle, float], float]):
    img = np.zeros((w, h, 3))
    img[:, :] = (0.5, 0.5, 0.5)
    from tqdm import tqdm
    for x in tqdm(range(w)):
        for y in range(h):
            # Draw circle boarder
            dist_from_target_center = math.sqrt(
                (x-c_target.x)**2 + (y-c_target.y)**2)
            if abs(dist_from_target_center - c_target.r) <= w*10**-3:
                img[y][x] = (0, 0, 1)
                continue
            dist_from_target_center = math.sqrt(
                (x-c_current.x)**2 + (y-c_current.y)**2)
            if abs(dist_from_target_center - (c_current.r)) <= w*10**-3:
                img[y][x] = (0.5, 0.5, 0.5)
                continue

            # Check if the point is within each circle
            taget_label = 1 if (c_target.x-x)**2 + \
                (c_target.y-y)**2 < c_target.r**2 else 0
            curr_label = 1 if (c_current.x-x)**2 + \
                (c_current.y-y)**2 < c_current.r**2 else 0

            err_level = err(x, y, c_current, q)

            if taget_label == curr_label:
                img[y][x] = scalar_to_color(1-err_level)
            else:
                img[y][x] = scalar_to_color(err_level)

    return img


def main(qs):
    # Build the image array
    # img = np.array([[(0.5, 0.5, 0.5) for x in range(w)] for y in range(h)])
    c_width = 0.3
    c_target = Circle(x=w//2+w//5, y=h//2, r=w*c_width)
    c_current = Circle(x=w//2-w//5, y=h//2, r=w*c_width)

    def err(x: int, y: int, circle: Circle, q: float):
        import math
        # Current circle
        dist_from_center = math.sqrt((x-circle.x)**2 + (y-circle.y)**2)
        if dist_from_center <= circle.r:
            p = dist_from_center / circle.r
            p = p**q
            return p/2
        elif dist_from_center < 2*circle.r:
            d_over_boarder = dist_from_center - circle.r
            p = d_over_boarder / circle.r
            rev_p = 1-p
            rev_p = rev_p ** q

            return (rev_p/2)
        else:
            return 0

    imgs = []
    for q in qs:
        img = build_img_array(w=w, h=h, c_target=c_target,
                              c_current=c_current, q=q, err=err)
        imgs.append(img)
    tag = "q_"
    for q in qs:
        tag += str(q) + "-"
    display_img(imgs, tag=tag + f"r_{c_width}-w_{w}-h_{h}")


if __name__ == "__main__":
    qs = [1, 10]
    main(qs)
