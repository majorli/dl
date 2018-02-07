import numpy as np
import csv
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from rcio import *

class Dataset:
    """Dataset"""
    _products = {}              # products in the dataset
    _customers = {}             # customers in the dataset
    _axis_p, _axis_c = [], []   # axis of products and customers
    _dates = set()              # dates in the dataset
    _num_data = 0               # number of non-zero data in the dataset
    _ds = None                  # dataset main table

    def __init__(self):
        return

    def generate_training_set(self, mask):
        assert(self._ds is not None)

        rc_state("Generating training set...")
        Y = self._ds.copy()
        nC = len(self._axis_c)
        nons = []
        nonsid = []
        for coord in range(nC):
            cid = self._axis_c[coord]
            if cid not in mask:
                # remember him, we'll remove him later
                nons.append(coord)
                nonsid.append(cid)
                continue
            msk = mask[cid]
            nan_p = []
            for pid in msk:
                try:
                    nan_p.append(self._axis_p.index(pid))
                except ValueError:
                    pass
            # end for
            # Y[nan_p, coord] = np.nan  # totally set to nan
            for coord_p in nan_p:
                if Y[coord_p, coord] == 0:
                    Y[coord_p, coord] = np.nan
        #end for
        Y = np.delete(Y, nons, axis=1)
        P = self._axis_p.copy()
        C = self._axis_c.copy()
        for cid in nonsid:
            C.remove(cid)
        ts = {
                "Y" : Y,
                "P" : P,
                "C" : C
                }
        rc_result("Training set generated. Removed {0:d} 'X-man' customers, found {1:d} data points to predict.".format(len(nons), np.sum(np.isnan(Y))))
        return ts


    def filter(self):
        _p_wl = []                  # whitelist of products
        _c_wl = []                  # whitelist of customers
        while True:
            rc_state(self.states())
            rc_highlight("You have six options to filter the dataset:")
            print("  1. Put/Remove given products into/from the whitelist to prevent them to be removed.")
            print("  2. Put/Remove given customers into/from the whitelist to prevent them to be removed.")
            print("  3. Remove given products.")
            print("  4. Remove given customers.")
            print("  5. Remove products that be sold by very few customers.")
            print("  6. Remove customers that sold very few kinds of products.")
            print("  7. Show current whitelists.")
            print("  8. Clear current whitelist of products.")
            print("  9. Clear current whitelist of customers.")
            print("  0. Exit filtering.")
            opt = rc_select("Now give me your choice (0..9): ", range_=range(10), t="highlight")
            if opt == 0:
                break
            rc_warn("REMEMBER: Whitelists are temporary, they will BE CLEARED after you leave filtering and they will NEVER BE SAVED!!!")
            if opt == 1 or opt == 2:
                ids = rc_input("Enter the ids to toggle whitelist states, separating by <spaces>: ").split()
                if opt == 1:
                    for i in ids:
                        si = i.zfill(8)
                        if si not in self._products:
                            print("Product id:", si, bcolors.FAIL + "not exists." + bcolors.ENDC)
                            continue
                        if si in _p_wl:
                            _p_wl.remove(si)
                            print("Product id:", si, ", name:", self._products[si], "is removed from whitelist.")
                        else:
                            _p_wl.append(si)
                            print("Product id:", si, ", name:", self._products[si], "is put into whitelist.")
                else:
                    for i in ids:
                        si = i.zfill(9)
                        if si not in self._customers:
                            print("Customer id:", si, bcolors.FAIL + "not exists." + bcolors.ENDC)
                            continue
                        if si in _c_wl:
                            _c_wl.remove(si)
                            print("Customer id:", si, ", name:", self._customers[si], "is removed from whitelist.")
                        else:
                            _c_wl.append(si)
                            print("Customer id:", si, ", name:", self._customers[si], "is put into whitelist.")
            elif opt == 3 or opt == 4:
                ids = rc_input("Enter the ids to remove from dataset, separating by <spaces>: ").split()
                if opt == 3:
                    for i in ids:
                        si = i.zfill(8)
                        if si not in self._products:
                            print("Product id:", si, bcolors.FAIL + "not exists." + bcolors.ENDC)
                            continue
                        if si in _p_wl:
                            print("Product id:", si, bcolors.FAIL + "is in the whitelist, cannot remove it." + bcolors.ENDC)
                            continue
                        coord = self._axis_p.index(si)
                        name = self._products.pop(si)
                        self._axis_p.remove(si)
                        self._ds = np.delete(self._ds, coord, axis=0)
                        print("Product id:", si, ", name:", name, "is removed from dataset.")
                else:
                    for i in ids:
                        si = i.zfill(9)
                        if si not in self._customers:
                            print("Customer id:", si, bcolors.FAIL + "not exists." + bcolors.ENDC)
                            continue
                        if si in _c_wl:
                            print("Customer id:", si, bcolors.FAIL + "is in the whitelist, cannot remove it." + bcolors.ENDC)
                            continue
                        coord = self._axis_c.index(si)
                        name = self._customers.pop(si)
                        self._axis_c.remove(si)
                        self._ds = np.delete(self._ds, coord, axis=1)
                        print("Customer id:", si, ", name:", name, "is removed from dataset.")
                self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            elif opt == 5 or opt == 6:
                cutoff = rc_select("Tell me the cutoff value (1.." + str(self._ds.shape[6-opt]) + "): ", range_=range(1, self._ds.shape[6-opt]))
                # removing
                d = np.nan_to_num(self._ds) > 0.0
                if opt == 5:
                    l = np.sum(d, axis=1) > cutoff
                    self._ds = self._ds[l, :]
                else:
                    l = np.sum(d, axis=0) > cutoff
                    self._ds = self._ds[:, l]
                l = ~l
                cnt = 0
                for i in range(len(l)-1, -1, -1):
                    if l[i]:
                        if opt == 5:
                            si = self._axis_p[i]
                            if si in _p_wl:
                                continue
                            _ = self._products.pop(si)
                            self._axis_p.remove(si)
                            cnt += 1
                        else:
                            si = self._axis_c[i]
                            if si in _c_wl:
                                continue
                            _ = self._customers.pop(si)
                            self._axis_c.remove(si)
                            cnt += 1
                rc_result("Totally {0} items removed.".format(np.sum(l)))
                self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            elif opt == 7:
                l = len(_p_wl)
                rc_result("Whitelist for products has {0} items:".format(l))
                for i in _p_wl:
                    print("  {0}[{1}]".format(self._products[i], i))
                l = len(_c_wl)
                rc_result("Whitelist for customers has {0} items:".format(l))
                for i in _c_wl:
                    print("  {0}[{1}]".format(self._customers[i], i))
            elif opt == 8:
                _p_wl = []
                rc_result("Whitelist of products cleared.")
            else:
                _c_wl = []
                rc_result("Whitelist of customers cleared.")
        # end of outer while loop
        return

    def browse(self):
        axis = rc_choose("Enter 'P' for browsing by products, or 'C' for browsing by customers: ", range_=["P", "C"])
        coords = []
        while len(coords) <= 0:
            coords = rc_input("Enter ids to browse, separating by <spaces>: ").split()
        zs = rc_ensure("Skip zeros? (Y/N) ")

        if axis == "P":
            for pid in coords:
                spid = pid.zfill(8)
                if spid not in self._products:
                    rc_fail("Product id: " + spid + " not exists.")
                    continue
                rc_result("Product id: " + spid + ", name: " + self._products[spid] + " weekly average sales:")
                x = self._axis_p.index(spid)
                dx = self._ds[x, :]
                for i in range(len(dx)):
                    if (dx[i] is np.nan or dx[i] == 0.0) and zs:
                        continue
                    c = "%9.4f by " % dx[i] + self._customers[self._axis_c[i]] + "[" + self._axis_c[i] + "]"
                    print(c)
                rc_state("This product has been sold by " + str(np.sum(dx > 0.0)) + " different customers")
        else:
            for cid in coords:
                scid = cid.zfill(9)
                if scid not in self._customers:
                    rc_fail("Customer id: " + scid + " not exists.")
                    continue
                rc_result("Customer id: " + scid + ", name: " + self._customers[scid] + " weekly average sales:")
                x = self._axis_c.index(scid)
                dx = self._ds[:, x]
                for i in range(len(dx)):
                    if (dx[i] is np.nan or dx[i] == 0.0) and zs:
                        continue
                    p = "%9.4f of " % dx[i] + self._products[self._axis_p[i]] + "[" + self._axis_p[i] + "]"
                    print(p)
                rc_state("This customer has sold " + str(np.sum(dx > 0.0)) + " kinds of products")
        return

    def plot(self):
        fig = plt.figure("Dataset")
        ax = fig.gca(projection="3d")
        X = np.arange(self._ds.shape[0])
        Y = np.arange(self._ds.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = np.nan_to_num(self._ds)[X, Y]
        surf = ax.plot_surface(X, Y, Z)
        plt.show()
        return

    def save(self, fn):
        np.savez("data/ds_" + fn, Mdata=[self._products, self._customers, self._dates], Ds=self._ds)

        with open("data/ds_" + fn + ".csv", "w", newline="") as f:
            f_csv = csv.writer(f)
            f_csv.writerow([" ", " ", " "] + self._axis_p)
            f_csv.writerow([" ", " ", " "] + [self._products[i] for i in self._axis_p])
            f_csv.writerow([" ", " ", " "] + list(np.nansum(self._ds > 0.0, axis=1)))
            count_by_cust = list(np.nansum(self._ds > 0.0, axis=0))
            nc = len(self._axis_c)
            for i in range(nc):
                c = self._axis_c[i]
                row = [c, self._customers[c], count_by_cust[i]] + list(self._ds[:, i])
                f_csv.writerow(row)
        return

    def __coords(self, axis, pids):
        pcoord = []
        for i in pids:
            pcoord.append(axis.index(i))
        return pcoord

    def load(self):
        """
        Load dataset from csv file
        """
        ds_type = rc_select("What kind of dataset do you want to load, (0) for raw records or (1) for saved dataset? ")
   
        if ds_type == 0:
            # csv
            dates = set()
            p, c = {}, {}
            r = []
            ## ds = None
            while True:
                fn = rc_getstr("Tell me the raw records filename (without ext '.csv'), enter nothing to stop loading raw records: ", keepblank=True, t="highlight")
                if fn == "":
                    break
                if not os.path.exists("raw/" + fn + ".csv"):
                    rc_fail("File not exists!")
                    continue
                len_o = len(r)
                f = open("raw/" + fn + ".csv")
                f_csv = csv.reader(f)
                headers = next(f_csv)
### 0:CUSTOMERSID, 1:DEPARTNAME, 2:ENTERPRISE, 3:PRINCIPAL, 4:PRODUCTNAME, 5:PRODUCTID, 6:BIZDATE, 7:SUM(A.DEFAULTQTY)
                rc_state("Loading sales records...")
                for rec in f_csv:
                    dates.add(rec[6].strip())
                    p[rec[5].strip()] = rec[4].strip()
                    cn = rec[2].strip()
                    if '*' in cn:
                        cn = rec[3].strip()
                    c[rec[0].strip()] = cn
                    r.append((rec[5].strip(), rec[0].strip(), float(rec[7].strip())))
                # end for
                f.close()
                len_f = len(r)
                rc_state("{0} raw sales records loaded this time, totally {1} records loaded.".format(len_f - len_o, len_f))

            if len(r) == 0:
                rc_fail("No sales records loaded!")
                return False

            rc_state("Generating products-customers sales table...")
            nP = len(p)
            nc = len(c)
            n = nP * nc
            self._products = p
            self._customers = c
            self._axis_p = sorted(p)
            self._axis_c = sorted(c)
            self._dates = dates
            self._ds = np.zeros((nP, nc))
            # self._products_to_predict = []
            px = sorted(p)
            pc = sorted(c)
            for rec in r:
                self._ds[px.index(rec[0]), pc.index(rec[1])] += rec[2]
            # end for
            days = len(dates)
            if days < 7:
                rc_warn("Hey, you only gave me {0} days' sale records. This dataset is hardly useful!".format(days))
            else:
                if days < 28:
                    rc_warn("{0} days' records, less than 4 weeks. Anyway, that's not such bad.".format(days))
                if days % 7 != 0:
                    rc_warn("{0} days' records, not integral multiple of one week. But don't mind, everything is okay.".format(days))
                    self._ds = self._ds * 7 / days
                else:
                    weeks = days // 7
                    rc_state("{0} days, i.e. {1} weeks' records. Perfect dataset!".format(days, weeks))
                    self._ds = self._ds / weeks
            self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            return True
        else:
            # npz
            fn = rc_getstr("What's the dataset name? Enter nothing to quit loading: ", keepblank=True, t="highlight")
            if fn == "":
                rc_warn("Quit.")
                return False
            if not os.path.exists("data/ds_" + fn + ".npz"):
                rc_fail("File not exists!")
                return False

            npz = np.load("data/ds_" + fn + ".npz")
            rc_state("Loading from early saved dataset...")
            # np.savez("ds_" + fn, Mdata=[self._products, self._customers, self._dates], Ds=self._ds)
            self._products = npz["Mdata"][0]
            self._customers = npz["Mdata"][1]
            self._axis_p = sorted(self._products)
            self._axis_c = sorted(self._customers)
            self._dates = npz["Mdata"][2]
            self._ds = npz["Ds"]
            self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            return True

    def states(self):
        s = "{num_products:d} products, {num_customers:d} customers, {num_data:d} non-zero data, dataset density = {data_density:.2f}%".format(num_products=len(self._axis_p), num_customers=len(self._axis_c), num_data=self._num_data, data_density=self.density() * 100)
        return s

    def density(self):
        return self._num_data / self._ds.shape[0] / self._ds.shape[1]

