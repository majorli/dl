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

    if dataset is not None and not rc_ensure("Load dataset will overwrite current dataset, are you sure? (Y/N) ", t="warn"):
        return

    dataset = rcd.Dataset()
    if dataset.load():
        rc_result("Okay! Dataset loaded successfully.")
        rc_result("The data density is {0:.2f}%. You can filter some products or customers that have very small number of sales to make the data density bigger.".format(dataset.density() * 100))
        if model is not None and g_mask is not None:
            model.fed(dataset.generate_training_set(g_mask))
            rc_result("Training set in the model is updated.")
    else:
        dataset = None
        rc_fail("Loading dataset failed or aborted!")

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
    fn = rc_getstr("Enter the dataset name, nothing to quit saving: ", keepblank=True)
    if fn != "":
        dataset.save(fn)
        rc_result("Saved Okay!")
    else:
        rc_warn("Quit.")
    return

def create_model():
    global dataset
    global model
    global g_mask

    rc_highlight("You can create a new model with hyperparameters from an early saved model.")
    ref = rc_getstr("Enter the ref-model name or nothing to create a new defalut model: ", keepblank=True)
    if ref != "" and not os.path.exists("model_" + ref + ".npz"):
        rc_fail("Where is this funny '{0}' model? But don't worry, I'll create a default one for you.".format(ref))
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
        if not rc_ensure("Load a new model will overwrite current model. Is it okay (Y/N)? ", t="warn"):
            return

    y = rc_getstr("Tell me the name of the model you want to load, nothing to quit loading: ", keepblank=True)
    if y == "":
        rc_warn("Quit.")
        return
    if not os.path.exists("data/model_" + y + ".npz"):
        rc_warn("Where is this funny '{0}' model?".format(y))
        return

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

    fn = rc_getstr("Tell me the name and I'll give you the mask, input nothing to quit loading: ", keepblank=True)
    if fn == "":
        rc_warn("Quit.")
        return
    
    fn = "data/mask_" + fn + ".json"
    if not os.path.exists(fn):
        rc_fail("Cannot find this mask!")
        return

    f = open(fn, "r")
    g_mask = json.load(f)
    f.close()

    model.fed(dataset.generate_training_set(g_mask))
    rc_result("Mask is put on the dataset, model ready. It's show time!")
    return

def _generate_one_round_mask(cust_cls):
    global g_mask

    fn = None
    while fn is None:
        fn = "raw/" + rc_getstr("Enter the filename of classes-products table (without .csv): ") + ".csv"
        if not os.path.exists(fn):
            rc_fail("File not found! Try again.")
            fn = None

    f = open(fn)
    f_csv = csv.reader(f)
    # 0:class, 1:productid, 2:[X], 3:[X], 4:[X], 5:mask, 6:[X]
    headers = next(f_csv)
    cls_mask = {}
    for rec in f_csv:
        try:
            m = int(rec[5].strip())
        except ValueError:
            m = 0
        if m == 0:
            c = rec[0].strip()
            p = rec[1].strip()
            if c in cls_mask:
                cls_mask[c].append(p)
            else:
                cls_mask[c] = [p]
            pass
        # end if
        pass
    # end for
    f.close()

    if g_mask is None:
        g_mask = {}
    for rec in cust_cls:
        p = rec[0].strip()      # customersid
        c = rec[3].strip()      # class
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

    # load customers-classes table
    fn = rc_getstr("Enter the filename of customers-classes table (without .csv), enter nothing to quit mask generating: ", t="highlight", keepblank=True)
    if fn == "":
        rc_warn("Quit.")
        return
    if not os.path.exists("raw/" + fn + ".csv"):
        rc_fail("File not exists!")
        return

    cust_cls = []
    f = open("raw/" + fn + ".csv")
    f_csv = csv.reader(f)
    _ = next(f_csv)
    for rec in f_csv:
        cust_cls.append(rec)
    f.close()
    if len(cust_cls) == 0:
        rc_fail("Error: Empty table.")
        return

    # generate round by round
    g_mask = None
    rounds = 0
    cont = True
    while cont:
        _generate_one_round_mask(cust_cls)
        rounds = rounds + 1
        cont =  rc_ensure("Week {0:d}: Done! Continue to another week (Y/N)?".format(rounds), t="highlight")

    # Save the mask
    fn = "data/mask_" + rc_getstr("Global mask needs be saved immediately. Give me a name: ") + ".json"
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
    c = rc_select(PROMPT, range_=range(len(menu)), t="highlight")
    if c == 0:
        break
    else:
        menu[c-1][1]()

# Script end here

