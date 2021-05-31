import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
from file_functions import *
import os
# plt.rcParams.update({'font.size': 16, 'font.family':'Times New Roman'})
plt.rcParams.update({'font.size': 16, 'font.family':'Computer Modern', 'text.usetex':True})


def standard_deviation(list_of_population):
    '''CALCULATES THE AVERAGE AND THE STANDARD DEVIATION OF A GIVEN LIST'''
    average = sum(list_of_population) / len(list_of_population)
    s = 0
    for i in range(len(list_of_population)):
        s += (list_of_population[i] - average)**2
    s = 1/(len(list_of_population) - 1) * s
    deviation = np.sqrt(s)
    return average, deviation

def adjust_yaxis(ax,ydif,v):
    '''SHIFT AXIS AX BY YDIFF, MAINTAINING POINT V AT THE SAME LOCATION'''
    inv = ax.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, ydif))
    miny, maxy = ax.get_ylim()
    miny, maxy = miny - v, maxy - v
    if -miny>maxy or (-miny==maxy and dy > 0):
        nminy = miny
        nmaxy = miny*(maxy+dy)/(miny+dy)
    else:
        nmaxy = maxy
        nminy = maxy*(miny+dy)/(maxy+dy)
    ax.set_ylim(nminy+v, nmaxy+v)

def align_yaxis(ax1, v1, ax2, v2):
    '''ADJUST AX2 YLIMIT SO THAT V2 IN AX2 IS ALIGNED TO V1 IN AX1'''
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    adjust_yaxis(ax2, (y1 - y2) / 2, v2)
    adjust_yaxis(ax1, (y2 - y1) / 2, v1)

def find_timestamp_at_slip(t, px1, px3, px5, slip_threshold):
    '''RETURNS A TIMESTAMP WHERE SYSTEM SLIPS'''
    index_at_slip = 0
    for i in range(len(t)):
        if abs(px1[i] - px1[0]) > slip_threshold and abs(px3[i] - px3[0]) > slip_threshold and abs(px5[i] - px5[0]) > slip_threshold: #    Slip threshold
            index_at_slip = i
            break
    # print(t[index_at_slip])
    return t[index_at_slip]

def find_moment_at_slip(x, y, t, px1, px3, px5, slip_threshold):
    '''RETURNS THE TIMESTAMP AND MOMENT AT SLIP'''
    index_at_slip = 0
    time = find_timestamp_at_slip(t, px1, px3, px5, slip_threshold)
    for i in range(len(x)):
        if x[i] > time:
            index_after_slip = i
            index_at_slip = i-1
            break
    mom = np.interp(time, [x[index_at_slip], x[index_after_slip]], [y[index_at_slip], y[index_after_slip]])
    # print(time, x[index_at_slip], x[index_after_slip], mom, y[index_at_slip], y[index_after_slip])
    return time, mom

def find_delay(momentfilename):
    '''RETURNS THE DELAY THAT IS NEEDED TO SHIFT THE MOMENT CURVE IN ORDER TO COMPENSATE FOR THE HARDWARE'''
    x, y, load, pressure, reaction = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    timestamps = []
    moms = []
    for i in range(1,len(reaction)):
        if int(reaction[i]) != int(reaction[i-1]):
            timestamps.append(x[i])
    for e in range(1, len(y)):
        if abs(y[e] - y[e-1]) > 0.03:
            moms.append(x[e-1])
            break
    error = abs(timestamps[0] - moms[0])
    return error

def collect_error(directory):
    '''RETURNS A LIST OF DELAYS FROM A DIRECTORY PATH'''
    list_of_error = []
    for filename in directory:
        error = find_delay(filename)
        list_of_error.append(error)
    return list_of_error

def output_corrected_moment(opencvfilename, momentfilename, slip_threshold):
    '''RETURNS THE MOMENT WHEN THE SYSTEM SLIPS, DISPLACEMENT DATA AND MOMENT DATA AS INPUT'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    x, y, load, pressure = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    for o in range(len(x)):
        x[o] = x[o] - 0.26

    time, mom = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold)
    return mom

def plot_my_graphs(opencvfilename, momentfilename, slip_threshold, export = False, outputNameFile = "tmp.pdf"):
    '''ONE PLOT WITH GRAPHS WITH DISPLACEMENT POINTS AND MOMENT GRAPH, INPUT DISPLACEMENT DATA AND MOMENT DATA'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    x, y, load, pressure, reaction = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    for o in range(len(x)):
        x[o] = x[o] - 0.26

    time, mom = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold)
    tmpName = outputNameFile.split('.')[0]
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName, '/')
            break
        elif element == '\\':
            tmpName = nofoldername(tmpName, '\\')
            break

    name = tmpName.split('.')[0]
    name = name + '_{}mm'.format(slip_threshold)
    print(name)
    f, ax1 = plt.subplots()
    ax1.spines['bottom'].set_color('#dddddd')
    ax1.spines['top'].set_color('#dddddd')
    ax1.spines['right'].set_color('red')
    ax1.spines['left'].set_color('red')
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')
    plt.title(name)
    f.set_figheight(5)
    f.set_figwidth(10)
    ax1.plot(x, y, 'b', label='Moment')
    ax1.plot([time], [mom], 'ro', markersize=2)
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_ylabel("Moment [Nm]")

    ax2.plot(t, x1 - x1[0], color='#15B01A', label='P0')
    ax2.plot(t, x2 - x2[0], color='#00FF00', label='P2')
    ax2.plot(t, x3 - x3[0], color='#AAFF32', label='P4')

    plt.axvline(x=time, color='r', linestyle='-.', label='Slip {}Nm'.format(mom.round(2)))
    ax1.set_xlabel("Time [s]")
    ax2.set_ylabel("x-displacements [mm]")
    plt.legend(loc='upper left')
    f.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        outputNameFile = 'pdfs/gammel_sealinxvac/' + name + '.png'
        plt.savefig(outputNameFile)
    else:
        plt.show()
    print(time, mom)

def plot_all_slip_graphs(opencvfilename, momentfilename, slip_threshold, export = False, outputNameFile = "tmp.pdf"):
    '''ONE PLOT WITH GRAPHS WITH DISPLACEMENT POINTS AND MOMENT GRAPH, INPUT DISPLACEMENT DATA AND MOMENT DATA'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    x, y, load, pressure = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    for o in range(len(x)):
        x[o] = x[o] - 0.26

    time1, mom1 = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold[0])
    time2, mom2 = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold[1])
    time3, mom3 = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold[2])

    tmpName = momentfilename
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName, '/')
            break
        elif element == '\\':
            tmpName = nofoldername(tmpName, '\\')
            break

    name = tmpName.split('.')[0]
    name = name + '_{}mm'.format(slip_threshold)
    print(name)
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    f.set_figheight(4)
    f.set_figwidth(6)
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')
    ax1.plot(x, y, 'b', label='Moment')

    '''POINTS THAT SHOW SLIP ON MOMENT GRAPH'''
    ax1.plot([time1], [mom1], 'ro', markersize=3)
    ax1.plot([time2], [mom2], 'ro', markersize=3)
    ax1.plot([time3], [mom3], 'ro', markersize=3)
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_ylabel("Moment [Nm]")

    '''DISPLACEMENT GRAPHS'''
    ax2.plot(t, x1 - x1[0], color='#15B01A', label='P0')
    ax2.plot(t, x2 - x2[0], color='#00FF00', label='P2')
    ax2.plot(t, x3 - x3[0], color='#AAFF32', label='P4')

    '''VERTICAL LINES AT SLIP'''
    plt.axvline(x=time1, color='r', linestyle='-.', lw=0.5, label='Slip {}Nm'.format(mom1.round(2)))
    plt.axvline(x=time2, color='r', linestyle='--', lw=0.5, label='Slip {}Nm'.format(mom2.round(2)))
    plt.axvline(x=time3, color='r', linestyle='-', lw=0.5, label='Slip {}Nm'.format(mom3.round(2)))
    ax1.set_xlabel("Time [s]")
    ax2.set_ylabel("x-displacements [mm]")
    plt.legend()
    plt.title(name)
    plt.xlim(0, time3 + 0.2)
    ax1.set_ylim(-0.3, mom3 + 5)
    ax2.set_ylim(0, 5)
    align_yaxis(ax1, 0, ax2, 0)
    f.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        outputNameFile = 'pdfs/gammel_sealinxvac/' + name + '.png'
        plt.savefig(outputNameFile)
    else:
        # plt.title(name)
        plt.show()

def plot_graphs_with_slip(opencvfilename, momentfilename, export=False):
    '''ONE PLOT WITH GRAPHS WITH DISPLACEMENT POINTS AND MOMENT GRAPH, INPUT DISPLACEMENT DATA AND MOMENT DATA'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    x, y, load, pressure = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    for o in range(len(x)):
        x[o] = x[o] - 0.26

    time = {}
    for i in range(1, len(x1)):
        if x1[i] - x1[0] != 0:
            time['P0'] = t[i-1]
            break
    for j in range(1, len(x2)):
        if x2[j] - x2[0] != 0:
            time['P2'] = t[j-1]
            break
    for k in range(1, len(x3)):
        if x3[k] - x3[0] != 0:
            time['P4'] = t[k-1]
            break

    time3, mom3 = find_moment_at_slip(x, y, t, x1, x2, x3, 3)

    tmpName = momentfilename
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName, '/')
            break
        elif element == '\\':
            tmpName = nofoldername(tmpName, '\\')
            break

    name = tmpName.split('.')[0]
    name = name + '_3mm'
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    f.set_figheight(4)
    f.set_figwidth(6)
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')
    ax1.plot(x, y, 'b', label='Moment')

    '''DISPLACEMENT GRAPHS'''
    ax2.plot(t, x1 - x1[0], color='#15B01A', label='P0')
    ax2.plot(t, x2 - x2[0], color='#00FF00', label='P2')
    ax2.plot(t, x3 - x3[0], color='#AAFF32', label='P4')

    '''POINTS THAT SHOW SLIP ON MOMENT GRAPH'''
    ax1.plot([time['P0']], [0], 'ro', markersize=3)
    ax1.plot([time['P2']], [0], 'ro', markersize=3)
    ax1.plot([time['P4']], [0], 'ro', markersize=3)
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_ylabel("Moment [Nm]")

    '''VERTICAL LINES AT SLIP'''
    plt.axvline(x=time['P0'], color='r', linestyle='-.', lw=0.5)
    plt.axvline(x=time['P2'], color='r', linestyle='--', lw=0.5)
    plt.axvline(x=time['P4'], color='r', linestyle='-', lw=0.5)

    ax1.set_xlabel("Time [s]")
    ax2.set_ylabel("x-displacements [mm]")
    plt.legend()
    plt.xlim(0, time3 + 0.2)
    ax1.set_ylim(-0.3, mom3)
    ax2.set_ylim(-0.3, mom3)
    # align_yaxis(ax1, 0, ax2, 0)
    # align_yaxis(ax1, 5, ax2, 5)
    # ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))




    '''Text'''
    plt.text(time['P0'], 5/4, 'Slip P0', ha='right', va='center', rotation=90)
    plt.text(time['P2'], 5/2, 'Slip P2', ha='right', va='center', rotation=90)
    plt.text(time['P4'], 3*5/4, 'Slip P4', ha='right', va='center', rotation=90)
    f.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        outputNameFile = 'pdfs/slip_points_' + name + '.svg'
        plt.savefig(outputNameFile)
    else:
        # plt.title(name)
        plt.show()

def plot_moments_old(momentfilename, export = False, outputNameFile = "tmp.pdf"):
    '''ONE PLOT WITH MOMENT CURVE'''
    x, y, load, pressure = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    tmpName = outputNameFile
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName)
            tmpName = tmpName.split('.')[0]
            break
    name = tmpName.split('.')[0]

    # plt.title(name)
    plt.plot(np.subtract(x, 0), y, 'b')
    plt.grid(b=None, which='major', axis='y')
    plt.xlabel("Time [s]")
    plt.ylabel("Moment [Nm]")
    plt.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        plt.savefig('pdfs/' + name + '_mom_only.png')
    plt.show()

def plot_double_reaction(momentfilename_1, momentfilename_2, export=False, filetype='png'):
    f_size = 20
    x1, y1, load1, pressure1, reaction1 = np.loadtxt(momentfilename_1, delimiter='\t', unpack=True)
    x2, y2, load2, pressure2, reaction2 = np.loadtxt(momentfilename_2, delimiter='\t', unpack=True)
    x_label_ax2 = np.arange(0, int(max(x2) + 1), 0.5)
    x_labels_ax2 = []
    for k in range(len(x_label_ax2)):
        if k == 0:
            pass
        if (k % 2) != 0:
            x_labels_ax2.append(' ')
            continue
        x_labels_ax2.append(str(x_label_ax2[k]))

    x_label_ax1 = np.arange(0, int(max(x1) + 1), 0.5)
    x_labels_ax1 = []
    for l in range(len(x_label_ax1)):
        if l == 0:
            pass
        if (l % 2) != 0:
            x_labels_ax1.append(' ')
            continue
        x_labels_ax1.append(str(x_label_ax1[l]))
    timestamps1 = []
    moms1 = []
    for i in range(1, len(reaction1)):
        if int(reaction1[i]) != int(reaction1[i - 1]):
            timestamps1.append(x1[i])
    for e in range(1, len(y1)):
        if abs(y1[e] - y1[e - 1]) > 0.03:
            moms1.append(x1[e - 1])
            break
    error1 = abs(timestamps1[0] - moms1[0])
    timestamps2 = []
    moms2 = []
    for j in range(1, len(reaction2)):
        if int(reaction2[j]) != int(reaction2[j - 1]):
            timestamps2.append(x2[j])
    for g in range(1, len(y2)):
        if abs(y2[g] - y2[g - 1]) > 0.03:
            moms2.append(x2[g - 1])
            break
    error2 = abs(timestamps2[0] - moms2[0])

    head_width1 = 0.045 * max(y1)
    head_length1 = 0.2 * head_width1
    head_width2 = 0.04 * max(y2)
    head_length2 = 0.15 * head_width2
    arrow_length = 0.5
    arrow_length2 = 1

    f, (ax1, ax2) = plt.subplots(1,2)
    f.set_figheight(7)
    f.set_figwidth(15)

    ax1.axvline(x=timestamps1[0], color='red', label='Moment applied')
    ax1.axvline(x=moms1[0], color='green', label='Moment registered')
    ax1.plot([timestamps1[0], moms1[0]], [max(y1), max(y1)], '-', color='black')
    ax1.arrow(timestamps1[0] - arrow_length * error1, max(y1), arrow_length * error1 - head_length1, 0, head_width=head_width1, head_length=head_length1, color='black')
    ax1.arrow(moms1[0] + arrow_length * error1, max(y1), -arrow_length * error1 + head_length1, 0, head_width=head_width1, head_length=head_length1, color='black')
    ax1.text(error1 / 2 + timestamps1[0], max(y1) - 0.1 * max(y1), "Delay {}s".format(error1.round(3)), ha='center', va='top',
             backgroundcolor='white', size=f_size)
    ax1.plot(x1, y1, 'b', label='Moment')
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_xlabel("Time [s]", fontsize=f_size)
    ax1.set_xticks(np.arange(0, int(max(x1) + 1), 0.5))
    ax1.set_ylabel("Moment [Nm]", fontsize=f_size)


    ax2.axvline(x=timestamps2[0], color='red', label='Moment applied')
    ax2.axvline(x=moms2[0], color='green', label='Moment registered')
    ax2.text(error2 / 2 + timestamps2[0], max(y2) - 0.1 * max(y2), "Delay {}s".format(error2.round(3)), ha='center',
             backgroundcolor='white', size=f_size, va='top')
    ax2.plot([timestamps2[0], moms2[0]], [max(y2), max(y2)], '-', color='black')
    ax2.arrow(timestamps2[0] - arrow_length2 * error2, max(y2),arrow_length2 * error2 - head_length2, 0, head_width=head_width2, head_length=head_length2, color='black')
    ax2.arrow(moms2[0] + arrow_length2 * error2, max(y2), -arrow_length2 * error2 + head_length2, 0, head_width=head_width2, head_length=head_length2, color='black')
    ax2.plot(x2, y2, 'b', label='Moment')
    ax2.grid(b=None, which='major', axis='y')
    ax2.set_xlabel("Time [s]", size=f_size)
    ax2.set_xticks(np.arange(0, int(max(x2) + 1), 0.5))
    ax2.legend(loc='lower right', fontsize=f_size)

    plt.sca(ax1)
    plt.xticks(size=f_size)
    plt.yticks(size=f_size)
    plt.xticks(np.arange(0, int(max(x1) + 1), 0.5), x_labels_ax1, size=f_size)
    plt.xlim(0, timestamps1[1])
    plt.sca(ax2)
    plt.yticks(size=f_size)
    plt.xticks(np.arange(0, int(max(x2) + 1), 0.5), x_labels_ax2, size=f_size)
    plt.xlim(0, timestamps2[1])

    f.tight_layout()
    if export == True:
        plt.savefig('pdfs/delay_plot_2.' + filetype.split('.')[-1])
    else:
        plt.show()

def plot_displacements(opencvfilename, export=False, outputNameFile="tmp.pdf"):
    '''ONE PLOT OF DISPLACEMENT OF 3 POINTS IN X DIRECTION'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    tmpName = opencvfilename
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName,'/')
            tmpName = tmpName.split('.')[0]
            break
        elif element == '\\':
            tmpName = nofoldername(tmpName, '\\')
            tmpName = tmpName.split('.')[0]
            break
    name = tmpName.split('.')[0]

    # plt.title(name)
    plt.plot(t, x1 - x1[0], label='P0')
    plt.plot(t, x2 - x2[0], label='P2')
    plt.plot(t, x3 - x3[0], label='P4')
    plt.grid(b=None, which='major', axis='y')
    plt.xlabel("Time [s]")
    plt.ylabel("Absolute Position [mm]")
    plt.legend()
    plt.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        plt.savefig('pdfs/' + name + '_displacement_plot.pdf')
    plt.show()

def slip_mom_all_tests(tracked_data_directory, txt_data_directory, slip_threshold):
    '''RETURNS A LIST OF THE AVERAGE AND STANDARD DEVIATION OF SLIP MOMENT FROM THE GIVEN LIST OF PATHS'''
    txt_dir = my_directories(txt_data_directory)
    trck_data_dir = my_directories(tracked_data_directory)
    big_lst = []
    avg_list = []
    s_list = []
    # txt_dir = my_directories('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload') #   My directories in the given path
    # trck_data_dir = my_directories('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload')
    if len(txt_dir) > len(trck_data_dir):
        exit("Number of directories for txt files are greater than tracked data directories")
    elif len(txt_dir) < len(trck_data_dir):
        exit("Number of directories for tracked data files are greater than txt file directories")
    for x in range(len(txt_dir)):
        txt_paths = list_with_paths(txt_dir[x]) #   My text paths in the given directory
        trck_data_paths = list_with_paths(trck_data_dir[x])
        if len(txt_paths) > len(trck_data_paths):
            exit("Number of txt files are greater than tracked data files in {}".format(txt_dir[x]))
        elif len(txt_paths) < len(trck_data_paths):
            print("Number of tracked data files are greater than txt files in {}".format(trck_data_dir[x]))
            exit()
        mom_lst = []
        for y in range(len(txt_paths)):
            mom = output_corrected_moment(trck_data_paths[y], txt_paths[y], slip_threshold)
            # print(mom)
            mom_lst.append(mom)
        big_lst.append(mom_lst)
    for z in range(len(big_lst)):
        avg, s = standard_deviation(big_lst[z])
        avg_list.append(avg)
        s_list.append(s)
    # print(avg_list, s_list)
    return avg_list, s_list

def change_order(list_1, origin_index):
    '''MOVES A GIVEN ELEMENT TO THE BACK OF THE LIST'''
    tmp = list_1[origin_index]
    list_1.pop(origin_index)
    list_1.append(tmp)

def sealinx_aload_bars(export=False):
    '''SLIP MOMENT IN THE SEALINX LINER WITH 0, 2 AND 5MM SLIP THRESHOLD, IN DIFFERENT AXIAL LOADS, BAR'''
    avg_list_0mm, s_list_0mm = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', 0)
    avg_list_2mm, s_list_2mm = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', 2)
    avg_list_5mm, s_list_5mm = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', 5)

    change_order(avg_list_0mm, 1)
    change_order(avg_list_2mm, 1)
    change_order(avg_list_5mm, 1)
    change_order(s_list_0mm, 1)
    change_order(s_list_2mm, 1)
    change_order(s_list_5mm, 1)

    width = 0.3
    fig, ax = plt.subplots()
    labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
    x = np.arange(len(labels))
    p1 = ax.bar(x - width, avg_list_0mm, width, yerr=s_list_0mm, label='0mm slip')
    p2 = ax.bar(x, avg_list_2mm, width, yerr=s_list_2mm, label='2mm slip')
    p3 = ax.bar(x + width, avg_list_5mm, width, yerr=s_list_5mm, label='5mm slip')
    ax.set_ylabel('Moment at slip [Nm]')
    ax.set_xlabel('Axial loads [kg]')
    ax.set_xticks(x)
    ax.legend(loc='upper left')
    ax.set_xticklabels(labels)
    plt.title("Seal-in-X liner, atmospheric pressure")
    fig.tight_layout()
    if export == True:
        plt.savefig('pdfs/aload_all.png')
    else:
        plt.show()

def sealinx_aload_plot_tuple(slip_threshold, export=False):
    avg_list_xl = []
    s_list_xl = []
    for i in range(len(slip_threshold)):
        avg_list, s_list = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', slip_threshold[i])
        change_order(avg_list, 1)
        change_order(s_list, 1)
        avg_list_xl.append(avg_list)
        s_list_xl.append(s_list)

    fig, ax = plt.subplots()
    labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
    x = np.arange(len(labels))
    for j in range(len(avg_list_xl)):
        plt.plot(x, avg_list_xl[j], 'o', ls='-', label='Slip {}mm'.format(slip_threshold[j]))
        ax.fill_between(x, np.subtract(avg_list_xl[j], s_list_xl[j]), np.add(avg_list_xl[j], s_list_xl[j]), alpha=0.2)
    ax.grid(b=None, which='major', axis='y')
    ax.set_ylabel('Moment at slip [Nm]')
    ax.set_xlabel('Axial loads [kg]')
    ax.set_xticks(x)
    ax.legend()
    ax.set_xticklabels(labels)
    # plt.title("Seal-in-X liner, atmospheric pressure")
    fig.tight_layout()
    if export == True:
        plt.savefig('pdfs/aload_all_2.png')
    else:
        plt.show()

def sealinx_aload_plot(slip_threshold, export=False):
    '''SLIP MOMENT IN THE SEALINX LINER WITH SINGLE SLIP THRESHOLD, IN DIFFERENT AXIAL LOADS, PLOT'''
    if type(slip_threshold) == tuple:
        sealinx_aload_plot_tuple(slip_threshold, export=export)
        exit()
    avg_list, s_list = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', slip_threshold)

    change_order(avg_list, 1)
    change_order(s_list, 1)

    fig, ax = plt.subplots()
    labels = ['10', '20', '30', '40', '50', '60', '70', '80', '90', '100']
    x = np.arange(len(labels))
    p1 = plt.plot(x, avg_list, 'o', ls='-')
    ax.fill_between(x, np.subtract(avg_list, s_list), np.add(avg_list, s_list), alpha=0.2)
    ax.grid(b=None, which='major', axis='y')
    ax.set_ylabel('Moment at slip [Nm]')
    ax.set_xlabel('Axial loads [kg]')
    ax.set_xticks(x)
    # ax.legend(loc='upper left')
    ax.set_xticklabels(labels)
    # plt.title("Seal-in-X liner, atmospheric pressure")
    fig.tight_layout()
    if export == True:
        plt.savefig('pdfs/aload_all_1.png')
    else:
        plt.show()

def sealinx_pressure_bars(export=False):
    '''SLIP MOMENT IN THE SEALINX LINER WITH 0, 2 AND 5MM SLIP THRESHOLD, IN DIFFERENT PRESSURE LEVELS, BAR'''
    avg_list_0mm, s_list_0mm = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure',
        'C:\\Users\\taal1\OneDrive - NTNU\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure', 0)
    avg_list_2mm, s_list_2mm = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure', 2)
    avg_list_5mm, s_list_5mm = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure', 5)

    change_order(avg_list_0mm, 0)
    change_order(avg_list_2mm, 0)
    change_order(avg_list_5mm, 0)
    change_order(s_list_0mm, 0)
    change_order(s_list_2mm, 0)
    change_order(s_list_5mm, 0)

    width = 0.3
    fig, ax = plt.subplots()
    labels = ['40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100']
    print(len(labels))
    x = np.arange(len(labels))
    p1 = ax.bar(x - width, avg_list_0mm, width, yerr=s_list_0mm, label='0mm slip')
    p2 = ax.bar(x, avg_list_2mm, width, yerr=s_list_2mm, label='2mm slip')
    p3 = ax.bar(x + width, avg_list_5mm, width, yerr=s_list_5mm, label='5mm slip')
    # plt.plot(x, avg_list_2mm, yerr=s_list_2mm)
    ax.set_ylabel('Moment at slip [Nm]')
    ax.set_xlabel('Pressure levels [kPa]')
    ax.set_xticks(x)
    ax.legend(loc='upper left')
    ax.set_xticklabels(labels)
    plt.title("Seal-in-X liner, 80kg axial load")
    fig.tight_layout()
    if export == True:
        plt.savefig('pdfs/pressure_levels_all.png')
    else:
        plt.show()

def sealinx_pressure_plot_tuple(slip_threshold, export=False):
    avg_list_xl = []
    s_list_xl = []
    for i in range(len(slip_threshold)):
        avg_list, s_list = slip_mom_all_tests('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure',
                                              'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure',
                                              slip_threshold[i])
        change_order(avg_list, 0)
        change_order(s_list, 0)
        avg_list_xl.append(avg_list)
        s_list_xl.append(s_list)

    fig, ax = plt.subplots()
    labels = ['40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100']
    x = np.arange(len(labels))
    for j in range(len(avg_list_xl)):
        plt.plot(x, avg_list_xl[j], 'o', ls='-', label='Slip {}mm'.format(slip_threshold[j]))
        ax.fill_between(x, np.subtract(avg_list_xl[j], s_list_xl[j]), np.add(avg_list_xl[j], s_list_xl[j]), alpha=0.2)
    ax.grid(b=None, which='major', axis='y')
    ax.set_ylabel('Moment at slip [Nm]')
    ax.set_xlabel('Pressure levels [kPa]')
    ax.set_xticks(x)
    ax.legend()
    ax.set_xticklabels(labels)
    fig.tight_layout()
    print('ey')
    if export == True:
        print('Thank you for purchasing 1000 gallons of milk!')
        plt.savefig('pdfs/pressure_levels_all_.png')
    else:
        plt.show()

def sealinx_pressure_plot(slip_threshold, export=False):
    '''SLIP MOMENT IN THE SEALINX LINER WITH 0, 2 AND 5MM SLIP THRESHOLD, IN DIFFERENT PRESSURE LEVELS, PLOT'''
    if type(slip_threshold) == tuple:
        sealinx_pressure_plot_tuple(slip_threshold, export=export)
        exit()
    avg_list, s_list = slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure',
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure', slip_threshold)
    change_order(avg_list, 0)
    change_order(s_list, 0)

    fig, ax = plt.subplots()
    labels = ['40', '45', '50', '55', '60', '65', '70', '75', '80', '85', '90', '95', '100']
    x = np.arange(len(labels))
    p1 = plt.plot(x, avg_list, 'o', ls='-')
    ax.fill_between(x, np.subtract(avg_list, s_list), np.add(avg_list, s_list), alpha=0.2)
    ax.grid(b=None, which='major', axis='y')
    ax.set_ylabel('Moment at slip [Nm]')
    ax.set_xlabel('Pressure levels [kPa]')
    ax.set_xticks(x)
    # ax.legend(loc='upper left')
    ax.set_xticklabels(labels)
    fig.tight_layout()
    if export == True:
        plt.savefig('pdfs/pressure_levels_all_1.png')
    else:
        plt.show()

def initial_results(slip_threshold, export=False):
    '''PLOT THAT COMPARES THE INITIAL TEST RESULTS'''
    avg_list, s_list= slip_mom_all_tests(
        'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\Ny_mock',
        'C:\\Users\\taal1\OneDrive - NTNU\Documents\\TMM4960\\Resultater\\Ny_mock', slip_threshold)

    labels = ['20', '40', '60', '80']
    x = np.arange(len(labels))
    plt.title("Comparison between different liners, {}mm slip, average".format(slip_threshold))
    plt.plot(x, avg_list[0:4],  '->', label='Dermo')
    plt.plot(x, avg_list[4:8], '->', label='Dermo w/sleeve')
    plt.plot(x, avg_list[8:12], '->', label='Dermo w/sleeve, np')
    plt.plot(x, avg_list[12:16], '->', label='Seal-in-X')
    plt.plot(x, avg_list[16:20], '->', label='Seal-in-X w/np')

    # plt.errorbar(x, avg_list[0:4], yerr=s_list[0:4], label='Dermo')
    # plt.errorbar(x, avg_list[4:8], yerr=s_list[4:8], label='Dermo w/sleeve')
    # plt.errorbar(x, avg_list[8:12], yerr=s_list[8:12], label='Dermo w/sleeve, np')
    # plt.errorbar(x, avg_list[12:16], yerr=s_list[12:16], label='Seal-in-X')
    # plt.errorbar(x, avg_list[16:20], yerr=s_list[16:20], label='Seal-in-X w/np')
    plt.grid(b=None, which='major', axis='y')
    plt.xlabel("Axial loads [kg]")
    plt.ylabel("Moment [Nm]")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    if export == True:
        name = 'pdfs/comparison_' + str(slip_threshold) + 'mm.png'
        plt.savefig(name)
    else:
        plt.show()

def plot_5_graphs(working_directory_opencv, working_directory_moments, slip_threshold, bool):
    '''PLOTS ALL GRAPHS FROM THE GIVEN PATHS'''
    # txtfiles = list_with_paths(working_directory_moments)
    # opencvfiles = list_with_paths(working_directory_opencv)
    if len(working_directory_opencv) != len(working_directory_moments):
        print("Uneven number of paths given!")
        exit()
    txtfiles = working_directory_moments
    opencvfiles = working_directory_opencv
    for i in range(len(txtfiles)):
        name = ''
        name = name + str(txtfiles[i]).split('\\')[-1] + '_{}mm.png'.format(slip_threshold)
        plot_my_graphs(opencvfiles[i], txtfiles[i], slip_threshold, export=bool, outputNameFile=name)

def avlast(txtfile_path, export = False):
    '''PLOT OF PRESSURE VS. AXIAL UNLOADING'''
    f_size = 30
    a, a, load, pressure = np.loadtxt(txtfile_path, delimiter='\t', unpack=True)
    f = plt.figure()
    f.set_figwidth(10)
    f.set_figheight(6)
    load = np.divide(load, 1000)
    pressure = np.divide(pressure, 1000)
    x_labels = ['100', '90', '80', '70', '60', '50', '40', '30', '20', '10', '0']
    plt.plot(np.flip(load), pressure, ls='-', lw=2)
    plt.grid(b=None, which='major', axis='y')
    plt.xlabel("Axial loads [kg]", fontsize = f_size)
    plt.ylabel("Pressure [kPa]", fontsize = f_size)
    plt.xticks(np.arange(0, 100 + 1, 10), x_labels, fontsize = f_size)
    plt.yticks(fontsize = f_size)
    f.tight_layout()
    if export == True:
        plt.savefig('pdfs/aload_vs_np_2.png')
    else:
        plt.show()

def pulse(opencvfilename, momentfilename, slip_threshold, export=False):
    '''PLOT PULSE GRAPH, MOMENT VS. DISPLACEMENTS OF THREE POINTS'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    x, y, load, pressure = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    for o in range(len(x)):
        x[o] = x[o] - 0.26

    time, mom = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold)

    f, ax1 = plt.subplots()
    ax1.spines['bottom'].set_color('#dddddd')
    ax1.spines['top'].set_color('#dddddd')
    ax1.spines['right'].set_color('red')
    ax1.spines['left'].set_color('red')
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')
    f.set_figheight(5)
    f.set_figwidth(10)
    ax1.plot(x, y, 'b', label='Moment', linewidth=1)
    ax1.plot([time], [mom], 'ro', markersize=2)
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_ylabel("Moment [Nm]")

    ax2.plot(t, np.subtract(x1, x1[0]), color='#15B01A', label='P0')
    ax2.plot(t, x2 - x2[0], color='#00FF00', label='P1')
    ax2.plot(t, x3 - x3[0], color='#AAFF32', label='P2')

    plt.axvline(x=time, color='r', linestyle='-', label='Slip {}Nm'.format(mom.round(2)))
    ax1.set_xlabel("Time [s]")
    ax2.set_ylabel("x displacements [mm]")
    plt.xlim(int(time) - 10, int(time) + 10)
    ax1.set_ylim(-0.5, max(y))
    ax2.set_ylim(-0.5, 0.5)
    plt.legend(loc='upper left')
    align_yaxis(ax1, 0, ax2, 0)
    # plt.title(name)
    f.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        outputNameFile = 'pdfs/cont_close_' + str(slip_threshold) + 'mm.png'
        plt.savefig(outputNameFile)
    else:
        plt.show()

def plot_std_err(type_liner, slip_threshold, export=False):
    '''SIMPLE PLOT WITH ERROR BARS, SINGLE PLOT'''
    p1 = 'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\std_avvik\\' + type_liner
    p2 = 'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\std_avvik\\' + type_liner
    avg_list, s_list= slip_mom_all_tests(p1, p2, slip_threshold)
    labels = ['20', '40', '60', '80']
    ylabels = ['-1', '0', '1', '2', '3', '4', '5', '6']
    x = np.arange(len(labels))
    y = np.arange(len(ylabels))
    plt.errorbar(x, avg_list, yerr=s_list)
    plt.grid(b=None, which='major', axis='y')
    plt.xlabel("Axial loads [kg]")
    plt.ylabel("Moment [Nm]")
    plt.xticks(x, labels)
    plt.yticks(y, ylabels)
    plt.tight_layout()
    name = type_liner + '_' + str(slip_threshold) + 'mm'
    if export == True:
        plt.savefig('pdfs/' + name + '.png')
    else:
        plt.show()

def get_avg_delay_from_tests(list_with_paths):
    '''RETURNS THE AVERAGE DELAY AND STANDARD DEVIATION OF A GIVEN LIST OF PATHS'''
    errors = collect_error(list_with_paths)
    avg, s = standard_deviation(errors)
    print(avg, s)

'''LOAD THE FOLLOWING DIRECTORIES TO DICTIONARIES'''
official_test_txt = group_txt_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\Ny_mock')
official_test_tracked = group_tracked_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\Ny_mock')
aload_test_txt = group_txt_paths_in_dictionary('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload')
aload_test_tracked = group_tracked_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload')
pressure_test_txt = group_txt_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure')
pressure_test_tracked = group_tracked_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\pressure')
reactions = group_txt_paths_in_dictionary('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\reaction')
official_test_txt_ext = group_txt_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.03\\txt')
official_test_tracked_ext = group_tracked_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.03\\csv')