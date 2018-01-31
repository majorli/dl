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
            menu.append(MENUITEMS[5])
            menu.append(MENUITEMS[6])
            menu.append(MENUITEMS[1])
            menu.append(MENUITEMS[2])
            menu.append(MENUITEMS[7])
            menu.append(MENUITEMS[8])
            menu.append(MENUITEMS[10])
            menu.append(MENUITEMS[9])

    menu.append(("Quit.", None))

    return menu

PROMPT = "Tell me your choice:> "

def load_dataset():
    global dataset
    global model
    global g_mask
    if dataset is not None:
        while True:
            y = rc_warn_in("Load dataset will overwrite current dataset, are you sure (Y/N)? ").upper()
            if y != "" and (y[0] == "Y" or y[0] == "N"):
                y = y[0]
                break
        if y == "N":
            return

    while True:
        y = rc_input("What kind of dataset do you want to load, (0) for raw records or (1) for saved dataset? ")[0]
        if y == "0" or y == "1":
            break
        else:
            rc_fail("WTF! Tell me just (0) for raw records or (1) for saved dataset? ")
    # end while
    
    if y == "0":
        fn = rc_input("Tell me the raw records csv filename (without ext '.csv'): ")
        fullname = fn + ".csv"
    else:
        fn = rc_input("What's the dataset name? ")
        fullname = "ds_" + fn + ".npz"      # filename of dataset: 'ds_' + name + ".npz"
    if os.path.exists(fullname):
        dataset = rcd.Dataset()
        if dataset.load(fn, int(y)):
            rc_result("Okay! Dataset '" + fn + "' is loaded.")
            den = round(dataset.density() * 100, 2)
            rc_result("The data density is " + str(den) + "%. You can filter some products or customers that have very small number of sales to make the data density bigger.")
            if model is not None and g_mask is not None:
                model.fed(dataset.generate_training_set(g_mask))
                rc_result("Training set in the model is updated.")
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
    fn = rc_input("Enter the dataset name, nothing to quit saving: ")
    if fn != "":
        dataset.save(fn)
        rc_result("Saved Okay!")
    else:
        rc_warn("Not saved.")
    return

def create_model():
    global dataset
    global model
    global g_mask

    rc_header("You can try a new model with hyperparameters from another early saved model.")
    ref = rc_input("Enter the ref-model name or nothing to create a new defalut model: ")
    if ref != "" and not os.path.exists("model_" + ref + ".npz"):
        rc_fail("Where is this funny '" + ref + "' model? But don't worry, I'll create a default one for you.")
        ref = ""
    model = rcm.Model(ref)
    # Feed dataset and mask. If no mask, send a warning.
    if g_mask is None:
        rc_warn("Model is created, but you need a mask to start training a model, load or generate one.")
    else:
        model.fed(dataset.generate_training_set(g_mask))
        rc_result("Done! Let's rock!")
    return

def load_model():
    global dataset
    global model
    global g_mask

    if model is not None:
        while True:
            y = rc_warn_in("Load a new model will overwrite current model. Is it okay (Y/N)? ").upper()
            if y != "" and (y[0] == "Y" or y[0] == "N"):
                y = y[0]
                break
        if y == "N":
            return

    while True:
        y = rc_input("Tell me the name of the model you want to load: ")
        if os.path.exists("model_" + y + ".npz"):
            break
        rc_warn("Where is this funny '" + y + "' model?")

    if model is None:
        model = rcm.Model("")
    model.load(y)
    rc_result("Model loaded successfully")
    return

def load_mask():
    global dataset
    global model
    global g_mask
    if dataset is None:
        rc_fail("You should first have a dataset, man!")
        return
    if model is None:
        rc_fail("You should first have a model, man!")
        return

    while True:
        fn = rc_input("Tell me the name and I'll give you the mask: ")
        if os.path.exists("mask_" + fn + ".json"):
            break

    fn = "mask_" + fn + ".json"
    f = open(fn, "r")
    g_mask = json.load(f)
    f.close()

    model.fed(dataset.generate_training_set(g_mask))
    rc_result("Mask is put on the dataset, model ready. It's show time!")
    return

def _generate_one_round_mask():
    global g_mask

    while True:
        fn = rc_input("Enter the filename of classes-products table (without .csv): ") + ".csv"
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
        fn = rc_input("Enter the filename of customers-classes table (without .csv): ") + ".csv"
        if os.path.exists(fn):
            break
        pass
    f = open(fn)
    f_csv = csv.reader(f)
    # 0:, 1:customersid, 2:customerno, 3:enterprise, 4:dicname(class), 5:isdemocustomer
    headers = next(f_csv)
    if g_mask is None:
        g_mask = {}
    for rec in f_csv:
        p = rec[1].strip()      # customersid
        c = rec[4].strip()      # class
        if c in cls_mask:
            if p in g_mask:
                g_mask[p] = [x for x in g_mask[p] if x in cls_mask[c]]  # intersection of original mask and current mask
                if len(g_mask[p]) == 0:
                    _ = g_mask.pop(p)
            else:
                g_mask[p] = cls_mask[c]         # no original mask, simply set to current mask
    # end for
    f.close()

    return

def generate_mask():
    global database
    global model
    global g_mask
    if dataset is None:
        rc_fail("You should first have a dataset, man!")
        return
    if model is None:
        rc_fail("You should first have a model, man!")
        return

    # generate round by round
    g_mask = None
    while True:
        _generate_one_round_mask()
        y = rc_highlight_in("Done. Say anything (e.g. 'y') to continue generating, or nothing to finish: ")
        if y == "":
            break

    # Save the mask
    while True:
        fn = rc_input("Global mask needs be saved immediately. Give me a name: ")
        if fn != "":
            break

    fn = "mask_" + fn + ".json"
    f = open(fn, "w")
    json.dump(g_mask, f)
    f.close()

    # Create the training set in current model
    model.fed(dataset.generate_training_set(g_mask))
    rc_result("Okay, the dataset along with this mask is fed into the model. Let's start!")
    return

def feed_model():
    global dataset
    global model
    global g_mask
    if model is None or dataset is None or g_mask is None:
        rc_fail("To feed a model, you should first have a model, a dataset, and a mask!")
        return
    model.fed(dataset.generate_training_set(g_mask))
    rc_result("Feed model by current dataset and mask, Okay.")
    return

def run_model():
    global model
    global dataset
    if model is None:
        rc_fail("You should first have a model, man!")
    else:
        if model.not_ready():
            rc_fail("You should first feed training set into the model.")
        else:
            model.run(dataset._products, dataset._customers)
    return

#def save_model():
#    global dataset
#    global model
#    if model is None:
#        rc_fail("You should first have a model, man!")
#        return
#
#    while True:
#        fn = rc_highlight_in("Enter the name of this model: ")
#        if fn != "":
#            break
#
#    model.save(fn)
#    rc_result("Save as '" + fn + "', Okay.")
#    return

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
    ("Run the model", run_model),
    ("Feed the model", feed_model)
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
    try:
        c = int(cmd)
        if c == 0:
            break;                      # break the main loop 'while True'
        if c < len(menu):
            menu[c-1][1]()              # call the defined function
            continue
    except ValueError:
        pass
    print(cmd)
    rc_fail("Are you kidding? Choose something from what I showed you.")
# end while

# Script end here

