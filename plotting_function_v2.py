import numpy as np
import os
import matplotlib.pyplot as plt
from file_functions import *
from plotting_functions import *



def send_to_csv_almm_v1(export=False):
    if export == False:
        exit('Did not export!')
    official_test_txt_ext = group_txt_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\txt')
    official_test_tracked_ext = group_tracked_paths_in_dictionary('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\csv')
    dermolistmom = []
    dermolisttrc =[]
    type = 'dermosleeve_vac'
    slip_threshold = [0, 1, 2, 3, 4, 5]
    with open('socket_mock_movements_1_dermo_sleeve_vac_isolated.txt', 'w') as fh:
        fh.write('Suspension\tmm\tTrial\tAW20kg\tAW40kg\tAW60kg\tAW80kg\n')
        for j in range(10):
            for k in range(1,5):
                moms = official_test_txt_ext[type + '_' + str(k * 20)][j]
                trc = official_test_tracked_ext[type + '_' + str(k * 20)][j]
                dermolistmom.append(moms)
                dermolisttrc.append(trc)
        momentpath = np.reshape(dermolistmom, (10,4))
        opencvpath = np.reshape(dermolisttrc, (10, 4))
        for i in range(len(slip_threshold)):
            for l in range(10):
                filename = name_split_my_file(momentpath[l, 0], '\\')
                AW20 = output_corrected_moment_v2(opencvpath[l, 0], momentpath[l, 0], slip_threshold[i])
                AW40 = output_corrected_moment_v2(opencvpath[l, 1], momentpath[l, 1], slip_threshold[i])
                AW60 = output_corrected_moment_v2(opencvpath[l, 2], momentpath[l, 2], slip_threshold[i])
                AW80 = output_corrected_moment_v2(opencvpath[l, 3], momentpath[l, 3], slip_threshold[i])
                # fh.write(f'{filename[0]}\t{slip_threshold[i]}\t{filename[-1]}\t{AW20}\t{AW40}\t{AW60}\t{AW80}\n')
                fh.write(f'{type}\t{slip_threshold[i]}\t{filename[-1]}\t{AW20}\t{AW40}\t{AW60}\t{AW80}\n')

def send_to_csv_almm_v2(export=False):
    if export == False:
        exit('Did not export!')
    official_test_txt_ext = group_txt_paths_in_dictionary(
        'C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\txt')
    official_test_tracked_ext = group_tracked_paths_in_dictionary(
        'C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\csv')
    n = list(official_test_tracked_ext.keys())
    slip_threshold = [0,1,2,3,4,5]
    with open('socket_mock_movements_2_dermo_sleeve_vac_isolated.txt', 'w') as fh:
        fh.write('Suspension\tmm\tTrial\tAW\tSlip moment\n')
        for element in n:
            for j in range(len(slip_threshold)):
                for i in range(len(official_test_txt_ext[element])):
                    filename = name_split_my_file(official_test_txt_ext[element][i], '\\')
                    mom = output_corrected_moment_v2(official_test_tracked_ext[element][i], official_test_txt_ext[element][i], slip_threshold[j])
                    if len(filename) > 3:
                        fh.write(f'{filename[0]}{filename[1]}\t{slip_threshold[j]}\t{filename[-1]}\t{filename[-2]}\t{mom}\n')
                    else:
                        fh.write(f'{filename[0]}\t{slip_threshold[j]}\t{filename[-1]}\t{filename[-2]}\t{mom}\n')

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

def output_corrected_moment_v2(opencvfilename, momentfilename, slip_threshold):
    '''RETURNS THE MOMENT WHEN THE SYSTEM SLIPS, DISPLACEMENT DATA AND MOMENT DATA AS INPUT, WITH COMPENSATION'''
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
    error = find_delay(momentfilename)
    for o in range(len(x)):
        x[o] = x[o] - error
    time, mom = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold)
    return mom

def plot_all_slip_graphs_v2(opencvfilename, momentfilename, slip_threshold, export = False):
    '''ONE PLOT WITH GRAPHS WITH DISPLACEMENT POINTS AND MOMENT GRAPH, INPUT DISPLACEMENT DATA AND MOMENT DATA'''
    data = np.genfromtxt(opencvfilename, delimiter=",",
                         names=["a", "Timestamp", "x1", "a", "x2", "a", "x3", "a"])
    x, y, load, pressure, reactions = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    t = data["Timestamp"]
    t = t[1:]
    x1 = data["x1"]
    x1 = x1[1:]
    x2 = data["x2"]
    x2 = x2[1:]
    x3 = data["x3"]
    x3 = x3[1:]
    delay = find_delay(momentfilename)
    for o in range(len(x)):
        x[o] = x[o] - delay

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
    print(name, delay)
    f, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    f.set_figheight(4)
    f.set_figwidth(6)
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')
    ax1.plot(x, y, 'b', label='Moment')
    lines = ['-.', '--', '-']
    counter = 0

    '''DISPLACEMENT GRAPHS'''
    ax2.plot(t, x1 - x1[0], color='#15B01A', label='P0')
    ax2.plot(t, x2 - x2[0], color='#00FF00', label='P2')
    ax2.plot(t, x3 - x3[0], color='#AAFF32', label='P4')

    if type(slip_threshold) == int:
        tmp = [slip_threshold]
        slip_threshold = tmp
    for element in slip_threshold:
        time1, mom1 = find_moment_at_slip(x, y, t, x1, x2, x3, element)
        '''POINTS THAT SHOW SLIP ON MOMENT GRAPH'''
        ax1.plot([time1], [mom1], 'ro', markersize=3)
        '''VERTICAL LINES AT SLIP'''
        plt.axvline(x=time1, color='r', ls=lines[counter], lw=0.5, label='Slip {}Nm'.format(mom1.round(2)))
        if counter == 2:
            counter = 0
        else:
            counter += 1
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_ylabel("Moment [Nm]")
    ax1.set_xlabel("Time [s]")
    ax2.set_ylabel("x-displacements [mm]")
    plt.legend()
    # plt.title(name)
    plt.xlim(0, time1 + 0.2)
    # plt.xticks([0.5, 1], ['0.5', '1'])
    ax1.set_ylim(0, 10)
    ax2.set_ylim(0, 5)
    # align_yaxis(ax1, 0, ax2, 0)
    ax2.set_yticks(np.linspace(ax2.get_yticks()[0], ax2.get_yticks()[-1], len(ax1.get_yticks())))
    f.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        outputNameFile = 'pdfs\\' + name + '.pdf'
        plt.savefig(outputNameFile)
    else:
        # plt.title(name)
        plt.show()

def plot_all_slip_graph_v2_all(list_opencvpath, list_momentpath, slip_threshold, export = False):
    if len(list_opencvpath) > len(list_momentpath):
        print('Number of OpenCV paths > Number of data paths')
        exit()
    elif len(list_opencvpath) < len(list_momentpath):
        print('Number of OpenCV paths < Number of data paths')
        exit()
    for i in range(len(list_opencvpath)):
        plot_all_slip_graphs_v2(list_opencvpath[i], list_momentpath[i], slip_threshold, export=export)

def plot_graphs_with_slip_v2(export=False):
    for i in range(1,5):
        for o in range(10):
            plot_all_slip_graphs_v2(official_test_tracked_ext['sealinxvac_' + str(i*20)][o], official_test_txt_ext['sealinxvac_' + str(i*20)][o], (1,2,3), export=export)

def aload_vs_disp(opencvfilename, momentfilename, slip_threshold, export = False, outputNameFile = "tmp.pdf"):
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
    error = find_delay(momentfilename)
    for o in range(len(x)):
        x[o] = x[o] - error

    time, mom = find_moment_at_slip(x, y, t, x1, x2, x3, slip_threshold)
    # tmpName = outputNameFile.split('.')[0]
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
    # print(name)
    f, ax1 = plt.subplots()
    ax1.spines['bottom'].set_color('#dddddd')
    ax1.spines['top'].set_color('#dddddd')
    ax1.spines['right'].set_color('red')
    ax1.spines['left'].set_color('red')
    ax2 = ax1.twinx()
    ax1.tick_params(axis='y', colors='blue')
    ax2.tick_params(axis='y', colors='green')
    # plt.title(name)
    f.set_figheight(7)
    f.set_figwidth(9)
    ax1.plot(x, load/1000, 'b', label='Axial load')
    # ax1.plot([time], [mom], 'ro', markersize=2)
    ax1.grid(b=None, which='major', axis='y')
    ax1.set_ylabel("Load [kg]")

    ax2.plot(t, x1 - x1[0], color='#15B01A', label='P0')
    ax2.plot(t, x2 - x2[0], color='#00FF00', label='P2')
    ax2.plot(t, x3 - x3[0], color='#AAFF32', label='P4')

    # plt.axvline(x=time, color='r', linestyle='-.', label='Slip {}Nm'.format(mom.round(2)))
    ax1.set_xlabel("Time [s]")
    ax2.set_ylabel("x-displacements [mm]")
    plt.legend(loc='lower right')
    ax1.set_ylim(0, 120)
    ax1.set_yticks(np.linspace(0, ax1.get_yticks()[-1], 13))
    ax2.set_ylim(0, 30)
    ax2label = ['0', '', '5', '', '10', '', '15', '', '20', '', '25', '', '30']
    ax2.set_yticks(np.linspace(0, ax2.get_yticks()[-1], len(ax1.get_yticks())), ax2label)
    plt.xlim(0, int(max(t)) + 1)
    f.tight_layout()
    if export == True:
        print("Thank you for buying 1000 liters of milk!")
        outputNameFile = 'pdfs/aload_vs_disp/' + name + '_aload_vs_disp.png'
        plt.savefig(outputNameFile)
    else:
        plt.show()
    print(time, mom)

def plot_reaction(momentfilename, export = False):
    '''ONE PLOT WITH MOMENT CURVE AND INITIAL LOADING VS REGISTERED LOADING'''
    f_size = 20
    x, y, load, pressure, reaction = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    x_label = np.arange(0, int(max(x) + 1), 0.5)
    x_labels = []
    for i in range(len(x_label)):
        if i == 0:
            pass
        if (i % 2) != 0:
            x_labels.append(' ')
            continue
        x_labels.append(str(x_label[i]))
    tmpName = momentfilename
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName, '/')
            tmpName = tmpName.split('.')[0]
            break
        elif element == '\\':
            tmpName = nofoldername(tmpName, '\\')
            tmpName = tmpName.split('.')[0]
            break
    name = tmpName.split('.')[0]
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
    print(timestamps, moms, error)
    f = plt.figure()
    axe = plt.axes()
    f.set_figheight(6)
    f.set_figwidth(9)
    plt.axvline(x=timestamps[0], color='red', label='Moment applied')
    plt.axvline(x=moms[0], color='green', label='Moment registered')
    plt.plot(x, y, 'b', label='Moment')

    plt.text(error/2 + timestamps[0], max(y) - 0.1*max(y), "Delay {}s".format(error.round(3)), ha='center', backgroundcolor='white', size=f_size)
    plt.grid(b=None, which='major', axis='y')
    plt.xlabel("Time [s]", size=f_size)
    plt.xticks(np.arange(0, int(max(x) + 1), 0.5), x_labels, size=f_size)
    plt.yticks(size=f_size)
    plt.ylabel("Moment [Nm]", size=f_size)
    plt.xlim(0, timestamps[1])
    print('xlim = ',axe.get_xlim(), 'ylim = ', axe.get_ylim())
    arrow_length = (axe.get_xlim()[1]) * 0.1
    arrow_head_width = (axe.get_ylim()[1]) * 0.035
    arrow_head_length = arrow_length * 0.15
    axe.arrow(timestamps[0] - arrow_length, max(y), arrow_length - arrow_head_length, 0, head_length= arrow_head_length, head_width= arrow_head_width, color='black')
    axe.arrow(moms[0] + arrow_length, max(y), -arrow_length + arrow_head_length, 0, head_width=arrow_head_width, head_length=arrow_head_length, color='black')
    plt.plot([timestamps[0] - arrow_length - arrow_head_length, moms[0] + arrow_length + arrow_head_length],[max(y), max(y)], '-', color='black')
    plt.legend(fontsize=f_size)
    f.tight_layout()
    if export == True:
        print("Thank you for purchasing 1000 gallons of milk!")
        plt.savefig('pdfs\\' + name + '.pdf')
    else:
        plt.show()

def plot_moments(momentfilename, export = False):   #   Trenger kanskje ikke denne
    '''ONE PLOT WITH MOMENT CURVE'''
    plt.rcParams.update({'font.size':20})
    x, y, load, pressure, reaction = np.loadtxt(momentfilename, delimiter='\t', unpack=True)
    tmpName = momentfilename
    for element in tmpName:
        if element == '/':
            tmpName = nofoldername(tmpName, '/')
            tmpName = tmpName.split('.')[0]
            break
        elif element == '\\':
            tmpName = nofoldername(tmpName, '\\')
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
        plt.savefig('pdfs/' + name + '_mom_only.pdf')
    else:
        plt.show()

def slip_mom_all_tests_v2(load, slip_threshold):
    txt_path_list, tracked_path_list, mom_list = [], [], []
    for element in official_test_tracked_ext:
        for i in range(len(official_test_tracked_ext[element])):
            if official_test_tracked_ext[element][i].split('_')[-4] == str(load):
                tracked_path_list.append(official_test_tracked_ext[element][i])
            if official_test_txt_ext[element][i].split('_')[-2] == str(load):
                txt_path_list.append(official_test_txt_ext[element][i])
    if len(tracked_path_list) != len(txt_path_list):
        exit('Uneven number of files!')
    for k in range(len(txt_path_list)):
        mom = output_corrected_moment_v2(tracked_path_list[k], txt_path_list[k], slip_threshold)
        mom_list.append(mom)
    tmp = np.reshape(list(official_test_txt_ext.keys()), (5,4))
    mom_list = np.reshape(mom_list, (5,10))
    if len(mom_list) != len(tmp):
        exit('Too many/few suspension systems')
    name = {}
    for m in range(len(tmp)):
        name[tmp[m, 0][:-3]] = mom_list[m]
    return name

def get_avg_std_all_ext_test(slip_threshold):
    avg_dict, s_dict = {}, {}
    for i in range(1,5):
        mom_dict = slip_mom_all_tests_v2(i * 20, slip_threshold)
        for element in mom_dict:
            liner = str(element)
            avg, s = standard_deviation(mom_dict[element])
            avg_dict[liner + '_' + str(i * 20)], s_dict[liner + '_' + str(i * 20)] = avg, s
    return avg_dict, s_dict

def get_avg_std_all_old_test(slip_threshold):
    avg_list, s_list = slip_mom_all_tests('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload',
                       'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', slip_threshold)
    change_order(avg_list, 1)
    change_order(s_list, 1)
    avg_dict, s_dict = {}, {}
    for i in range(1, 11):
        avg_dict['sealinx_' + str(i * 10)] = avg_list[i - 1]
        s_dict['sealinx_' + str(i * 10)] = s_list[i - 1]
    return avg_dict, s_dict

def plot_delay_vs_individual(slip_threshold, export=False):
    old_avg_dict, old_s_dict = get_avg_std_all_old_test(slip_threshold)
    new_avg_dict, new_s_dict = get_avg_std_all_ext_test(slip_threshold)
    names = ['sealinx_20', 'sealinx_40', 'sealinx_60', 'sealinx_80']
    latex = ['sealinx\_20', 'sealinx\_40', 'sealinx\_60', 'sealinx\_80']
    old_avg, new_avg, old_s, new_s = [], [], [], []
    xticks = np.arange(1, len(names) + 1)
    fig = plt.figure()
    for element in names:
        old_avg.append(old_avg_dict[element])
        new_avg.append(new_avg_dict[element])
        old_s.append(old_s_dict[element])
        new_s.append(new_s_dict[element])
    plt.plot(xticks, old_avg, 'o', ls='-', label='Constant shift (Average)')
    plt.plot(xticks, new_avg, 'o', ls='-', label='Individual shift')
    plt.xticks(xticks, latex)
    plt.yticks(np.arange(6))
    plt.grid(b=None, which='major', axis='y')
    plt.ylabel('Moment of Force [Nm]')
    plt.xlabel('Suspension method')
    plt.xlim(0.5, len(names) + 0.5)
    plt.legend()
    plt.tight_layout()
    if export == True:
        plt.savefig('pdfs\\avg_shift_vs_ind_shift_' + str(slip_threshold) + 'mm.png')
    else:
        plt.show()

def plot_suspension_methods(slip_threshold, export=False):
    avg_dict, s_dict = get_avg_std_all_ext_test(slip_threshold)
    new_dict = {}
    new_dict['dermo'], new_dict['dermosleeve'], new_dict['dermosleeve_vac'], new_dict['sealinx'], new_dict['sealinxvac'] = [], [], [], [], []
    for element in avg_dict:
        if element[:-3] == 'dermo':
            new_dict['dermo'].append(avg_dict[element])
        elif element[:-3] == 'dermosleeve':
            new_dict['dermosleeve'].append(avg_dict[element])
        elif element[:-3] == 'dermosleeve_vac':
            new_dict['dermosleeve_vac'].append(avg_dict[element])
        elif element[:-3] == 'sealinx':
            new_dict['sealinx'].append(avg_dict[element])
        elif element[:-3] == 'sealinxvac':
            new_dict['sealinxvac'].append(avg_dict[element])
    x = [20, 40, 60, 80]

    fig = plt.figure()
    names = ['Dermo', 'Dermo + sleeve', 'Dermo + sleeve + negative pressure', 'Seal-in-X', 'Seal-in-X + negative pressure']
    for key in new_dict:
        plt.plot(x, new_dict[key], 'o', ls='-')
    plt.ylabel('Moment at slip [Nm]')
    plt.xticks(np.arange(1, len(x) + 1) * 20, x)
    plt.xlabel('Axial loads [kg]')
    plt.grid(b=None, which='major', axis='y')
    plt.legend(names, loc='lower left', bbox_to_anchor=(0,1.,1.,0.2), mode='expand', ncol=2, fontsize=9)
    plt.tight_layout()
    if export == True:
        plt.savefig('pdfs\\suspension_method_all_' + str(slip_threshold) + 'mm.pdf')
    else:
        plt.show()

def plot_graph_with_avg_std(liner_type, slip_threshold, export=False):
    avg_dict, s_dict = get_avg_std_all_ext_test(slip_threshold)
    name = str(liner_type)
    new_avg_dict = {}
    new_s_dict = {}
    new_avg_dict[name] = []
    new_s_dict[name] = []
    for avg in avg_dict:
        if avg[:-3] == name:
            new_avg_dict[name].append(avg_dict[avg])
    for s in s_dict:
        if s[:-3] == name:
            new_s_dict[name].append(s_dict[s])
    xlabels = np.linspace(20,80,4)
    stdmin = np.subtract(new_avg_dict[name], new_s_dict[name])
    stdmax = np.add(new_avg_dict[name], new_s_dict[name])
    plt.plot(xlabels, new_avg_dict[name], 'o', ls='-', color='red')
    plt.fill_between(xlabels, stdmin, stdmax, alpha=0.3,color='red')
    plt.ylim(0, int(max(stdmax)) + 1.5)
    plt.xticks(xlabels)
    plt.ylabel('Moment at Slip [Nm]')
    plt.xlabel('Axial Loads [kg]')
    plt.grid(b=None, which='major', axis='y')
    plt.tight_layout()
    if export == True:
        plt.savefig('pdfs\\avg_std_' + name + '_' + str(slip_threshold) + 'mm.pdf')
        print('Saved')
    else:
        plt.show()

def whisker_plot_mom_vs_susp(load, slip_threshold, export=False):
    mom_dict = slip_mom_all_tests_v2(load, slip_threshold)
    fig, axe = plt.subplots()
    xticks = np.arange(1,6)
    counter = 1
    names = ['Dermo', 'Dermosleeve', 'Dermosleeve Vac', 'Seal-in-X', 'Seal-in-X Vac']
    for element in mom_dict:
        axe.boxplot(mom_dict[element], positions=[counter])
        counter += 1
    plt.xticks(xticks, names, size=12)
    plt.ylabel('Moment of Force [Nm]')
    plt.grid(b=None, which='major', axis='y')
    plt.tight_layout()
    if export == True:
        plt.savefig('pdfs/comparing_graph_' + str(load) + 'kg_' + str(slip_threshold) + 'mm.pdf')
        print('Saved')
    else:
        plt.show()