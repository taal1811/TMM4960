import numpy as np
import os

def list_with_paths(my_path):
    '''RETURNS A LIST OF CSV PATHS FROM A GIVEN DIRECTORY PATH'''
    momentdata_path = []
    for (dirpath, dirnames, filenames) in os.walk(my_path):
        filenames.sort()
        for names in filenames:
            momentdata_txt = os.path.join(dirpath, names)
            momentdata_path.append(momentdata_txt)
    return momentdata_path

def my_directories(my_path):
    '''RETURNS A LIST OF DIRECTORY NAMES FROM THE GIVEN DIRECTORY PATH'''
    directories = []
    for (dirpath, dirnames, filenames) in os.walk(my_path):
        dirnames.sort()
        for i in range(len(dirnames)):
            directories.append(os.path.join(dirpath, dirnames[i]))
    return directories

def nofoldername(path, delimeter):
    '''RETURNS THE LAST STRING FORM THE GIVEN DELIMETER'''
    fileName = path.split(delimeter)[-1]
    return fileName

def name_split_my_file(path, delimeter):
    '''RETURNS A NAME'''
    filename = nofoldername(path, delimeter)
    filename = filename.split('.')[0]
    filename = filename.split('_')
    return filename

def path_delimeter_type(string):
    '''RETURNS THE DELIMETER IN A GIVEN STRING/PATH'''
    for letter in string:
        if letter == '\\':
            delimeter = '\\'
        elif letter == '/':
            delimeter = '/'
    return delimeter

def group_txt_paths_in_dictionary(folderpath_string):
    '''GROUPS THE FILEPATHS TOGETHER, NAMED AFTER THE FOLDER NAMES. DIRECTORY NAMES SHOULD BE VERY SIMILAR TO THE FILENAMES'''
    txt_paths_list = list_with_paths(folderpath_string)
    dirlist = my_directories(folderpath_string)
    if len(dirlist) == 0:
        print("No folders in the given path. Sort your .txt files into folders!")
        exit()
    for i in range(len(dirlist)):
        d1 = path_delimeter_type(dirlist[i])
        dirlist[i] = dirlist[i].split(d1)[-1]
    mydict = {}
    for dir in dirlist:
        mydict[dir] = []
        for paths in txt_paths_list:
            d_temp = path_delimeter_type(paths)
            if paths.split(d_temp)[-1].split('.')[0][:-2] == dir:
                mydict[dir].append(paths)
    return mydict

def group_tracked_paths_in_dictionary(folderpath_string):
    '''GROUPS THE TRACKED FILEPATHS TOGETHER, NAMED AFTER THE FOLDER NAMES. DIRECTORY NAMES SHOULD BE VERY SIMILAR TO THE FILENAMES'''
    txt_paths_list = list_with_paths(folderpath_string)
    dirlist = my_directories(folderpath_string)
    if len(dirlist) == 0:
        print("No folders in the given path. Sort your .txt files into folders!")
        exit()
    for i in range(len(dirlist)):
        d1 = path_delimeter_type(dirlist[i])
        dirlist[i] = dirlist[i].split(d1)[-1]
    mydict = {}
    # print(dirlist)
    for dir in dirlist:
        mydict[dir] = []
        for paths in txt_paths_list:
            d_temp = path_delimeter_type(paths)
            if paths.split(d_temp)[-1].split('.')[0][:-26] == dir:
                mydict[dir].append(paths)
    return mydict