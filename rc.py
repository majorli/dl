# Huzhou Tobacco Corp. Products recommendation system
#
# Libin, 201801

import numpy as np
import os
import csv
import json

import rcds as rcd
import rcmodel as rcm
from rcio import *

# Global data
dataset = None
model = None
g_mask = None       # {customer_1 : [product, ..., product], ..., customer_m : [product, ..., product]}

## Utilities
def __create_menu():
    global dataset
    menu = []
    if dataset is None:
        menu.append(MENUITEMS[0])
        menu.append(MENUITEMS[1])
        menu.append(MENUITEMS[7])
        menu.append(MENUITEMS[8])
    else:
        if model is None:
            menu.append(MENUITEMS[0])
            menu.append(MENUITEMS[3])
            menu.append(MENUITEMS[4])
            menu.append(MENUITEMS[5])
            menu.append(MENUITEMS[6])
            menu.append(MENUITEMS[1])
            menu.append(MENUITEMS[2])
        else:
            menu.append(MENUITEMS[0])
            menu.append(MENUITEMS[3])
            menu.append(MENUITEMS[4])
            menu.append(MENUITEMS[6])
            menu.append(MENUITEMS[1])
            menu.append(MENUITEMS[2])
            menu.append(MENUITEMS[7])
            menu.append(MENUITEMS[8])
            menu.append(MENUITEMS[9])
            menu.append(MENUITEMS[10])
            menu.append(MENUITEMS[11])
            menu.append(MENUITEMS[12])

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
    global g_mask

    rc_header("You can try a new model with hyperparameters from another early saved model.")
    ref = rc_input("Enter the ref-model name or nothing to create a new defalut model: ")
    if ref != '' and not os.path.exists(ref + ".npz"):
        rc_fail("Where is this funny '" + ref + "' model? But don't worry, I'll create a default one for you.")
        ref = ""
    model = rcm.Model(ref)
    # Feed dataset and mask. If no mask, send a warning.
    if g_mask is None:
        rc_warn("Model is created, but you need a mask to start training a model, load or generate one.")
    else:
        model.fed(dataset, g_mask)
        rc_result("Done! Let's rock!")
    return

## In a saved model: hyperparameters (num_features, algorithm, learning_rate, L2)
def load_model():
    global dataset
    global model
    pass
    return

def load_mask():
    global dataset
    global model
    global g_mask
    while True:
        fn = rc_input("Tell me the name and I'll give the mask: ")
        if fn != "" and os.path.exists(fn + ".json"):
            break

    fn += ".json"
    f = open(fn, "r")
    g_mask = json.load(f)
    f.close()

    model.fed(dataset, g_mask)
    rc_result("Mask is put on the dataset, model ready. It's show time!")
    return

def generate_mask():
    global database
    global model
    global g_mask
    while True:
        fn = rc_input("Enter the filename of classes-products table (without .csv): ").strip() + ".csv"
        if os.path.exists(fn):
            break
        pass
    f = open(fn)
    f_csv = csv.reader(f)
    # 0:, 1:class, 2:productsid, 3:[X], 4:[X], 5:[X], 6:mask, 7:[X]
    headers = next(f_csv)
    cls_mask = {}
    for rec in f_csv:
        try:
            m = int(rec[6].strip())
        except ValueError:
            m = 0
        if m == 0:
            c = rec[1].strip()
            p = rec[2].strip()
            if c in cls_mask:
                cls_mask[c].append(p)
            else:
                cls_mask[c] = [p]
            pass
        # end if
        pass
    # end for
    f.close()

    while True:
        fn = rc_input("Enter the filename of customers-classes table (without .csv): ").strip() + ".csv"
        if os.path.exists(fn):
            break
        pass
    f = open(fn)
    f_csv = csv.reader(f)
    # 0:, 1:customersid, 2:customerno, 3:enterprise, 4:dicname(class), 5:isdemocustomer
    headers = next(f_csv)
    g_mask = {}
    for rec in f_csv:
        p = rec[1].strip()      # customersid
        c = rec[4].strip()      # class
        if c in cls_mask:
            g_mask[p] = cls_mask[c]
        pass
    # end for
    f.close()

    # Save the mask
    while True:
        fn = rc_input("Global mask needs be saved immediately. Give me a filename: ")
        if fn != "":
            break

    fn += ".json"
    f = open(fn, "w")
    json.dump(g_mask, f)
    f.close()

    # Create the training set in current model
    model.fed(dataset, g_mask)
    rc_result("Okay, the dataset along with this mask is fed into the model. Let's start!")

    return

# def save_mask():
#     global dataset
#     global model
#     global g_mask
#     while True:
#         fn = rc_input("Give me a filename to save the mask: ")
#         if fn != "":
#             break
# 
#     fn += ".json"
#     f = open(fn, "w")
#     json.dump(g_mask, f)
#     f.close()
#     rc_result("Saved Okay!")
#     return

def train_model():
    global dataset
    global model
    pass
    return

def save_model():
    global dataset
    global model
    pass
    return

def export_results():
    global dataset
    global model
    pass
    return

def cluster_features():
    global dataset
    global model
    pass
    return

MENUITEMS = [
    ("Load a dataset.", load_dataset),
    ("Load a model.", load_model),
    ("Create a model.", create_model),
    ("Browse dataset.", browse_dataset),
    ("Plot dataset.", plot_dataset),
    ("Filter dataset.", filter_dataset),
    ("Save dataset.", save_dataset),
    ("Load mask", load_mask),
    ("Generate mask", generate_mask),
    ("Train the model", train_model),
    ("Save the model", save_model),
    ("Cluster features", cluster_features),
    ("Export results", export_results)
]

## Script start here

rc_header("Hello, here is the PRODUCT RECOMMENDATION SYSTEM for HUZHOU TOBACCO CORP.")

# Main loop
while True:
    if dataset is None:
        rc_warn("Dataset: None")
    else:
        rc_state("Dataset: " + dataset.states())
    if model is None:
        rc_warn("Model: None")
    else:
        rc_state("Model: " + model.states())
    if g_mask is None:
        rc_warn("Mask: None")
    else:
        rc_state("Mask: Okay")

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

