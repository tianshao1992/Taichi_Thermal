import taichi as ti
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
from matplotlib import ticker, rcParams

"""data-read"""
case = 3
load = [[800, 1000], [1000, 1200], [1200, 1400]]

nelt, nely, nelz0, nelz1, nelz2 = int(3600./0.1)+1, 31, 5, 15, 5
# nelt, nely, nelz0, nelz1, nelz2 = int(3600./0.01)+1, 61, 11, 31, 11
nelz = nelz0 + nelz1 + nelz2 - 2

lam0, lam1, lam2 = 0.85, 0.0312, 0.85
rho0, rho1, rho2 = 1610, 374, 1610
cp0, cp1, cp2 = 780, 1010, 780

nelt_m, nelz_m, nely_m = 3601, 3, 3
measure = np.loadtxt('data\\meas' + str(case) + '.txt')
measure = measure.astype(np.float32).reshape((-1, nelz_m, nely_m))[:, ::-1, :].transpose((0, 2, 1))


node_y = np.linspace(0.0, 0.3, nely)
node_z_0 = np.linspace(0.0, 0.002, nelz0)
node_z_1 = np.linspace(0.002, 0.03, nelz1)
node_z_2 = np.linspace(0.03, 0.032, nelz2)
node_t = np.linspace(0.0, 3600., nelt)
load_y = np.linspace(load[case-1][0], load[case-1][1], nely)

"""data-preparation"""
node_z = np.concatenate((2*node_z_0[0:1] - node_z_0[1:2],
                         node_z_0, node_z_1[1:], node_z_2[1:],
                         2*node_z_2[-1:] - node_z_2[-2:-1]) , axis=0)[None, :]
node_y = np.concatenate((2*node_y[0:1] - node_y[1:2], node_y, 2*node_y[-1:] - node_y[-2:-1]) , axis=0)[:, None]
node_y_ = np.tile(node_y, (1, nelz+2))  # Ny x Nz
node_z_ = np.tile(node_z, (nely+2, 1))  # Ny x Nz
delta_y = node_y_[1:, :] - node_y_[:-1, :]
delta_z = node_z_[:, 1:] - node_z_[:, :-1]
delta_t = 3600. / (nelt - 1)

import scipy.interpolate as interpolate
int_y = interpolate.interp1d(np.array([node_y[1,0], node_y[int(nely/2)+1,0], node_y[-2,0]]), measure, axis=1, kind='slinear')
meas_ = int_y(node_y[1:-1,0])
int_t = interpolate.interp1d(np.linspace(0., 3600., nelt_m), meas_, axis=0, kind='linear')
meas_ = int_t(node_t)

alph_y_ = np.ones((nely+1, nelz+2), dtype=np.float32)
alph_z_ = np.ones((nely+2, nelz+1), dtype=np.float32)
alph_y_[:, :nelz0] = lam0 / rho0 / cp0
alph_y_[:, nelz0] = lam0 * lam1 / (lam0 + lam1) / ((rho0 + rho1) / 2 * (cp0 + cp1) / 2)
alph_y_[:, nelz0+1:-(nelz2+1)] = lam1 / rho1 / cp1
alph_y_[:, -(nelz2+1)] = lam1 * lam2 / (lam1 + lam2) / ((rho1 + rho2) / 2 * (cp1 + cp2) / 2)
alph_y_[:, -nelz2:] = lam2 / rho2 / cp2
alph_y_ /= delta_y ** 2

alph_z_[:, :nelz0] = lam0 / rho0 / cp0
alph_z_[:, nelz0:-nelz2] = lam1 / rho1 / cp1
alph_z_[:, -nelz2:] = lam2 / rho2 / cp2
alph_z_ /= delta_z ** 2

temp = np.ones_like(node_y_) * 20
temp[1:-1, -2] = load_y
solution = np.zeros((nelt, nely, nelz), dtype=np.float32)
solution[0] = temp[1:-1, 1:-1]

"""taichi-preparation"""
ti.init(arch=ti.cpu, default_fp=ti.f32, default_ip=ti.i32, cpu_max_num_threads=12, debug=False)
ti_dtype = ti.f32

node_y = ti.field(ti_dtype, shape=(nely+2, nelz+2))
node_z = ti.field(ti_dtype, shape=(nely+2, nelz+2))
temp_0 = ti.field(ti_dtype, shape=(nely+2, nelz+2))  # derivative of compliance
temp_1 = ti.field(ti_dtype, shape=(nely+2, nelz+2))  # derivative of compliance
alph_y = ti.field(ti_dtype, shape=(nely+1, nelz+2))  # derivative of compliance
alph_z = ti.field(ti_dtype, shape=(nely+2, nelz+1))  # derivative of compliance
meas_t = ti.field(ti_dtype, shape=(nelt, nely, nelz_m))  # derivative of compliance

node_y.from_numpy(node_y_)
node_z.from_numpy(node_z_)
alph_y.from_numpy(alph_y_)
alph_z.from_numpy(alph_z_)
temp_0.from_numpy(temp)
meas_t.from_numpy(meas_)
# temp_1.from_numpy(temp)

@ti.kernel
def forward():

    for i, j in temp_0:
        if (i > 0 and j > 0 and i < nely + 1 and j < nelz):

            temp_1[i, j] = temp_0[i, j] + delta_t * (
                           alph_y[i, j] * (temp_0[i + 1, j] - temp_0[i, j]) +
                           alph_y[i - 1, j] * (temp_0[i - 1, j] - temp_0[i, j]) +
                           alph_z[i, j] * (temp_0[i, j + 1] - temp_0[i, j]) +
                           alph_z[i, j - 1] * (temp_0[i, j - 1] - temp_0[i, j]))
        elif (j == nelz):
            temp_1[i, j] = temp_0[i, j]

#
# @ti.kernel
# def forward():
#
#     for i, j in temp_0:
#         if (i > 0 and j > 0 and i < nely + 1 and j < nelz):
#
#             temp_1[i, j] = temp_0[i, j] + delta_t * (
#                            alph_y[i + 1, j] * (temp_0[i + 1, j] - temp_0[i, j]) / (node_y[i+1, j]-node_y[i, j]) ** 2 +
#                            alph_y[i, j - 1] * (temp_0[i - 1, j] - temp_0[i, j]) / (node_y[i, j]-node_y[i-1, j]) ** 2 +
#                            alph_z[i, j + 1] * (temp_0[i, j + 1] - temp_0[i, j]) / (node_z[i, j+1]-node_z[i, j]) ** 2 +
#                            alph_z[i, j - 1] * (temp_0[i, j - 1] - temp_0[i, j]) / (node_z[i, j]-node_z[i, j-1]) ** 2)
#         elif (j == nelz):
#             temp_1[i, j] = temp_0[i, j]


@ti.kernel
def update(t: ti.int32):
    for i, j in temp_0:
        if (i == 0):
            temp_1[i, j] = temp_1[i+2, j]
        if (i == nely + 1):
            temp_1[i, j] = temp_1[i-2, j]

        if (j == 0):
            temp_1[i, j] = temp_1[i, j + 2]
        elif (j == nelz0):
            if (i>0 and i<nely + 1):
                temp_1[i, j] = meas_t[t, i-1, 0]
        elif (j == int(nelz/2)+1):
            if (i > 0 and i < nely + 1):
                temp_1[i, j] = meas_t[t, i-1, 1]
        elif (j == nelz-nelz2+1):
            if (i > 0 and i < nely + 1):
                temp_1[i, j] = meas_t[t, i-1, 2]

        temp_0[i, j] = temp_1[i, j]

def caculate():
    for t in range(1, nelt):

        forward()
        solution[t] = temp_1.to_numpy()[1:-1, 1:-1]
        update(t)

        # print(i)



if __name__ == '__main__':

    import os
    import scipy.io as sio

    name = 'thermal-2d-t-taichi'
    work_path = os.path.join('work', name, str(case))
    isCreated = os.path.exists(work_path)
    if not isCreated:
        os.makedirs(work_path)

    sta_time = time.time()

    print('start caculate!')
    caculate()
    print('cost time: {:.2e}'.format(time.time() - sta_time))

    try:
        sio.savemat(os.path.join(work_path, 'res.mat'), {'solution': solution})
    except:
        pass

    """visualization"""
    t1 = solution[:, (0, int(nely / 2), -1), nelz0-1]
    t2 = solution[:, (0, int(nely / 2), -1), int(nelz / 2)]
    t3 = solution[:, (0, int(nely / 2), -1), -nelz2]
    measure_visual = np.stack((t1, t2, t3), axis=2)
    vis_step = 90

    plt.rc('font', family='Times New Roman', size=25)
    font1 = {'family': 'Times New Roman',
             'style': 'normal',
             'size': 25,
             }
    fig = plt.figure(1, figsize=[20, 20])
    ax = fig.add_subplot(111)
    ax.axis('off')
    gs = gridspec.GridSpec(3, 3)
    gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.3, hspace=0.3)

    for i in range(3):
        for j in range(3):
            ax = plt.subplot(gs[i, j])
            ax.plot(node_t, measure_visual[:, j, 2-i])
            ax.scatter(np.linspace(0.0, 3600, 3601)[::vis_step], measure[::vis_step, j, 2-i], c='r')
            ax.set_xlim(-100, 3700)
            # ax.set_xlim(0, )
            ax.set_xlabel('t/s', font=font1)
            ax.set_ylabel('T/℃', font=font1)

            ax.legend(['pred', 'exp'], loc='best', prop=font1)

    plt.savefig(os.path.join(work_path, 'measure.svg'))

    time_ind = [1, 2, 5, 10, 100, 500, 1000, 2000, 3600]
    vis_step = int((nelt - 1) / 3600)
    solution_visual = solution[::vis_step]


    fig = plt.figure(2, figsize=[24, 15])
    ax = fig.add_subplot(111)
    ax.axis('off')
    gs = gridspec.GridSpec(3, 3)
    gs.update(top=0.95, bottom=0.07, left=0.1, right=0.9, wspace=0.3, hspace=0.3)

    for i in range(len(time_ind)):
        # plt.subplot(3, 3, i + 1)
        ax = plt.subplot(gs[i//3, i%3])

        h = ax.contour(node_y_[1:-1, 1:-1]*1000, node_z_[1:-1, 1:-1]*1000, solution_visual[time_ind[i]],
                      levels=20, linestyles='-', linewidths=0.5, colors='k')

        h = ax.pcolormesh(node_y_[1:-1, 1:-1]*1000, node_z_[1:-1, 1:-1]*1000, solution_visual[time_ind[i]],
                          cmap='RdYlBu_r', shading='gouraud', antialiased=True, snap=True)

        # divider = make_axes_locatable(ax)
        # cax = divider.append_axes("right", size="5%", pad=0.1, ylabel='T/℃')
        # # ax.axis('equal')

        cb = fig.colorbar(h, extend='both', shrink=1, label='T/℃', pad=0.01)
        ticker_locator = ticker.MaxNLocator(nbins=8)
        cb.locator = ticker_locator; cb.update_ticks()
        # cb.set_ticks(np.linspace(0, 1400, 8))
        h.set_clim(vmin=20, vmax=load[case-1][1])
        cb.ax.tick_params(axis='both', which='both', length=1)
        cb.ax.set_ylabel('T/℃')


        ax.set_xlabel('x/mm', font=font1)
        ax.set_ylabel('y/mm', font=font1)
        ax.set_title('t = ' + str(time_ind[i]) + 's', font=font1)

        # plt.title(str(time_ind[i]))
        # plt.colorbar()
        # plt.clim(vmin=20, vmax=load[case-1][1])
    # plt.savefig(res_path + 'field_' + str(t) + '-' + str(iter) + '.jpg')
    plt.savefig(os.path.join(work_path, 'snapshot.jpg'), dpi=200)

