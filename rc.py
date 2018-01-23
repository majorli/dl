# -*- coding: GBK -*-

# Huzhou Tobacco Corp. Products recommendation system
#
# Libin, 201801

import numpy as np
import os

import rcds as rcd
import rcmodel as rcm

# Global data
dataset = None
model = None

## Utilities
def __create_menu():
    global dataset
    menu = []
    if dataset is None:
        if model is None:
            menu.append(MENUITEMS[0])
            menu.append(MENUITEMS[1])
        else:
            pass
    else:
        if model is None:
            menu.append(MENUITEMS[0])
            menu.append(MENUITEMS[1])
            menu.append(MENUITEMS[2])
            menu.append(MENUITEMS[3])
            menu.append(MENUITEMS[4])
            menu.append(MENUITEMS[5])
            menu.append(MENUITEMS[6])
        else:
            pass

    menu.append(("Quit.", None))

    return menu

def load_dataset():
    global dataset
    y = input("Load dataset will overwrite current dataset, are you sure? (Y/N) ").strip()[0].upper()
    if y != "Y":
        return

    while True:
        y = input("What kind of dataset do you want to load, (0) for raw records or (1) for saved dataset? ").strip()[0]
        if y == "0" or y == "1":
            break
        else:
            print(bcolors.FAIL + "WTF! Tell me just (0) for raw records or (1) for saved dataset? " + bcolors.ENDC)
    # end while
    
    fn = input("Tell me the database name (extname is not needed, I know them): ").strip()
    fullname = fn
    if y == "0":
        fullname += ".csv"
    else:
        fullname += ".npz"
    if os.path.exists(fullname):
        dataset = rcd.Dataset()
        if dataset.load(fn, int(y)):
            print(bcolors.OKGREEN + "Okay! Dataset", dataset._name, "is loaded." + bcolors.ENDC)
            den = round(dataset.density() * 100, 2)
            print(bcolors.OKGREEN + "The data density is " + str(den) + "%. You can filter some products or customers that have very small number of sales to make the data density bigger." + bcolors.ENDC)
        else:
            dataset = None
            print(bcolors.FAIL + "This is not a valid dataset!" + bcolors.ENDC)
    else:
        print(bcolors.FAIL + "Hey! There's not such a dataset!" + bcolors.ENDC)

    return

def browse_dataset():
    global dataset
    dataset.browse()
    return

def plot_dataset():
    global dataset
    dataset.plot()
    return

def filter_dataset():
    global dataset
    dataset.filter()
    return

def save_dataset():
    global dataset
    fn = input("Enter a filename: ").strip()
    dataset.save(fn)
    print(bcolors.OKGREEN + "Saved Okay!" + bcolors.ENDC)
    return

MENUITEMS = [
    ("Load a dataset.", load_dataset),
    ("Load a model.", rcm.open_model),
    ("Try a new model.", rcm.create_model),
    ("Browse dataset.", browse_dataset),
    ("Plot dataset.", plot_dataset),
    ("Filter dataset.", filter_dataset),
    ("Save dataset.", save_dataset)
]

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

PROMPT = "Tell me your choice:> "

## Script start here

print(bcolors.HEADER + "Hello, here is the PRODUCT RECOMMENDATION SYSTEM for HUZHOU TOBACCO CORP." + bcolors.ENDC)

# Main loop
while True:
    if dataset is None:
        print(bcolors.OKBLUE + "Dataset: None" + bcolors.ENDC)
    else:
        print(bcolors.OKBLUE + "Dataset: " + dataset.states() + bcolors.ENDC)
    if model is None:
        print(bcolors.OKBLUE + "Model: None" + bcolors.ENDC)
    else:
        print(bcolors.OKBLUE + "Model: " + bcolors.ENDC)

    print("Now you can do these things:")
    menu = __create_menu()
    for i in range(len(menu) - 1):
        print(str(i+1) + ": " + menu[i][0])
    print("0: " + menu[-1][0])
    cmd = input(PROMPT).strip()
    if len(cmd) > 0 and cmd[0].isdigit():
        c = int(cmd[0])
        if c == 0:
            break;                      # break the main loop 'while True'
        if c < len(menu):
            menu[c-1][1]()              # call the defined function
            continue
    print(bcolors.FAIL + "Are you kidding? Choose something from what I showed you." + bcolors.ENDC)
# end while


# Script end here

