# Huzhou Tobacco Corp. Products recommendation system
#
# Libin, 201801

import numpy as np
import os

import rcds as rcd
import rcmodel as rcm
from rcio import *

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
            pass
    else:
        if model is None:
            menu.append(MENUITEMS[0])
            menu.append(MENUITEMS[3])
            menu.append(MENUITEMS[4])
            menu.append(MENUITEMS[5])
            menu.append(MENUITEMS[6])
            menu.append(MENUITEMS[1])
            menu.append(MENUITEMS[2])
            pass
        else:
            menu.append(MENUITEMS[0])
            menu.append(MENUITEMS[3])
            menu.append(MENUITEMS[4])
            menu.append(MENUITEMS[6])
            menu.append(MENUITEMS[1])
            menu.append(MENUITEMS[2])
            menu.append(MENUITEMS[7])

    menu.append(("Quit.", None))

    return menu

PROMPT = "Tell me your choice:> "

def load_dataset():
    global dataset
    y = rc_warn_in("Load dataset will overwrite current dataset, are you sure? (Y/N) ")[0].upper()
    if y != "Y":
        return

    while True:
        y = rc_input("What kind of dataset do you want to load, (0) for raw records or (1) for saved dataset? ")[0]
        if y == "0" or y == "1":
            break
        else:
            rc_fail("WTF! Tell me just (0) for raw records or (1) for saved dataset? ")
            pass
        pass
    # end while
    
    fn = rc_input("Tell me the database name (extname is not needed, I know them): ")
    fullname = fn
    if y == "0":
        fullname += ".csv"
    else:
        fullname += ".npz"
    if os.path.exists(fullname):
        dataset = rcd.Dataset()
        if dataset.load(fn, int(y)):
            rc_result("Okay! Dataset " + dataset._name + " is loaded.")
            den = round(dataset.density() * 100, 2)
            rc_result("The data density is " + str(den) + "%. You can filter some products or customers that have very small number of sales to make the data density bigger.")
        else:
            dataset = None
            rc_fail("This is not a valid dataset!")
    else:
        rc_fail("Hey! There's not such a dataset!")

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
    fn = rc_input("Enter a filename: ")
    dataset.save(fn)
    rc_result("Saved Okay!")
    return

def create_model():
    global dataset
    global model
    rc_header("You can try a new model with hyperparameters from another early saved model.")
    ref = rc_input("Enter the ref-model name or nothing to create a new defalut model: ")
    if ref != '' and not os.path.exists(ref + ".mat"):
        rc_fail("Where is this funny '" + ref + "' model? But don't worry, I'll create a default one for you.")
        ref = ""
    model = rcm.Model(ref)
    rc_result("Done! Let's rock!")
    return

def load_model():
    global dataset
    global model
    pass
    return

def generate_mask():
    global dataset
    global model
    pass
    return

MENUITEMS = [
    ("Load a dataset.", load_dataset),
    ("Load a model.", load_model),
    ("Try a new model.", create_model),
    ("Browse dataset.", browse_dataset),
    ("Plot dataset.", plot_dataset),
    ("Filter dataset.", filter_dataset),
    ("Save dataset.", save_dataset),
    ("Generate mask", generate_mask)
]

## Script start here

rc_header("Hello, here is the PRODUCT RECOMMENDATION SYSTEM for HUZHOU TOBACCO CORP.")

# Main loop
while True:
    if dataset is None:
        rc_state("Dataset: None")
    else:
        rc_state("Dataset: " + dataset.states())
    if model is None:
        rc_state("Model: None")
    else:
        rc_state("Model: " + model.states())

    rc_highlight("Now you can do these things:")
    menu = __create_menu()
    for i in range(len(menu) - 1):
        print(str(i+1) + ": " + menu[i][0])
        pass
    print("0: " + menu[-1][0])
    cmd = rc_highlight_in(PROMPT)
    if len(cmd) > 0 and cmd[0].isdigit():
        c = int(cmd[0])
        if c == 0:
            break;                      # break the main loop 'while True'
        if c < len(menu):
            menu[c-1][1]()              # call the defined function
            continue
    rc_fail("Are you kidding? Choose something from what I showed you.")
    pass
# end while


# Script end here

