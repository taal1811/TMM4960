from plotting_functions import *
from plotting_function_v2 import *
from file_functions import *

if __name__ == '__main__':
    # print(pressure_test_txt.keys())
    # print(pressure_test_txt['sealinxvac_80_100'])
    # plot_all_slip_graphs(aload_test_tracked['sealinxvac_' + str(i*20)][o], aload_test_txt['sealinxvac_' + str(i*20)][o], (0,1.5,3), export=True)
    # plot_all_slip_graph_v2_all(official_test_tracked_ext['sealinx_20'], official_test_txt_ext['sealinx_20'], (1,2,3), False)
    # plot_all_slip_graphs_v2(official_test_tracked_ext['sealinx_80'][2], official_test_txt_ext['sealinx_80'][2],(0, 1.5, 3), export=True)
    # liner = 'dermosleeve_vac_80'
    # for i in range(len(official_test_txt_ext[liner])):
    #     aload_vs_disp(official_test_tracked_ext[liner][i], official_test_txt_ext[liner][i], 1, export=True)
    # plot_all_slip_graphs_v2('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\csv\\sealinxvac_20_0_tracked_coordinates.csv', 'C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\txt\\sealinxvac_20_0.txt',1)

    # plot_graph_with_avg_std('sealinx', 1.5, export=False)

    # print(get_avg_std_all_test('80', 1))
    # whisker_plot_mom_vs_susp(80, 1.5, export=True)
    # plot_suspension_methods(1, export=True)
    # plot_moments_old('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.07\\sealinx-80-4.txt', export=False)

    # plot_graphs_with_slip(aload_test_tracked['sealinx_80'][1], aload_test_txt['sealinx_80'][1], export=False) # Bilde som viser slippunkter i paper (0mm)
    # sealinx_aload_plot_tuple((0,1.5,3), export=False)
    # sealinx_pressure_plot(1 ,export=False)
    # plot_reaction(reactions['reaction_quick'][2], export=False)
    # plot_moments(official_test_txt_ext['sealinx_80'][6], export=True)
    # plot_reaction('2021.05.03/txt/sealinxvac_20/sealinxvac_20_2.txt')
    # plot_reaction('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.03\\txt\\sealinxvac_20\\sealinxvac_20_1.txt')
    # plot_double_reaction(reactions['reaction_quick'][4], reactions['reaction_heavy'][4], export=False, filetype='.svg')
    # print(name_split_my_file(official_test_txt['sealinx_80'][1], '\\'))
    # send_to_csv(official_test_tracked_ext['sealinx_80'], official_test_txt_ext['sealinx_80'])
    # send_to_csv_almm_v1('tmp1.txt', 'dermo', 'w', export=True)
    # send_to_csv_almm_v2(export=False)
    # plot_mom_vs_susp(80, 0)
    # send_to_csv_almm_v1_all(zzz, export=False)
    # plot_displacements(reactions['reaction_heavy'][2])
    # plot_displacements(official_test_tracked_ext['sealinx_80'][6], export=True)
    # print(slip_mom_all_tests_v2(80,1))
    # avg, s = get_avg_std_all_ext_test(1)
    # print(slip_mom_all_tests('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\aload', 'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\aload', 1))
    # print(get_avg_std_all_old_test(1))
    # plot_delay_vs_individual(1, export=False)

    # n = list(official_test_txt_ext.keys())
    # for element in n:
    #     print(len(official_test_tracked_ext[element]), len(official_test_txt_ext[element]))
    #     print('--')
    # get_avg_delay_from_tests(reactions['reaction_heavy'])
    # get_avg_delay_from_tests(reactions['reaction_quick'])
    # plot_moments('2021.05.10\\txt\\dermosleeve_vac_80_0.txt')


    # pulse('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_80_pulse_1_autoTracked_coordinates.csv', 'C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_80_pulse_1.txt', 0, export=False)
    # pulse('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_80_cont_1_autoTracked_coordinates.csv', 'C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_80_cont_1.txt', 0, export=False)
    # sealinx_aload_bars()
    # sealinx_pressure_plot(3, export=False)
    # sealinx_aload_plot(3, export=False)
    # initial_results(5, export=False)
    # plot_5_graphs(official_test_tracked['dermosleeve_vac_40'], official_test_txt['dermosleeve_vac_40'], 0, False)
    # avlast('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_100_avlasting.txt', export=False)

    # plot_std_err('dermo', 2, export=False)

    # avg_list_0mm, s_list_0mm = slip_mom_all_tests(
    #     'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure',
    #     'C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.04.15\\sealinx_pressure', 0)
    # print(list_with_paths('C:\\Users\\taal1\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\tracked_data\\sealinx_pressure'))


    # plot_displacements('tracked_data/sealinx_80_pulse_1_autoTracked_coordinates.csv', export=False, outputNameFile="pdfs/sealinx_80kg_pulse.pdf")
    # plot_my_graphs('cont_tests/sealinx_80_cont_2_tracked_coordinates.csv', '2021.04.07/sealinx-80-cont-2.txt', export=True, outputNameFile="pdfs/sealinx-80kg_cont.pdf")
    # plot_my_graphs('tracked_data/sealinxvac_80_1_autoTracked_coordinates.csv', '2021.04.15/sealinxvac_80_1.txt', export=False, outputNameFile="pdfs/sealinxvac_80kg_1.pdf")

    # for i in range(1,5):
    #     os.mkdir('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\csv\\dermosleeve_vac_' + str(i * 20))
    #     os.mkdir('C:\\Users\\taal1\\OneDrive - NTNU\\Documents\\TMM4960\\Resultater\\2021.05.10\\txt\\dermosleeve_vac_' + str(i * 20))
    print('Programme finished')