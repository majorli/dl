import numpy as np
import csv
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from rcio import *

class Dataset:
    """Dataset"""
    _name = None                # raw data file, csv, sales records
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
            Y[nan_p, coord] = np.nan
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
        rc_result("Training set generated. Removed " + str(len(nons)) + " 'X-man' customers, found " + str(np.sum(np.isnan(Y))) + " data points to predict.")
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
            print("  5. Remove products that be saled by very few customers.")
            print("  6. Remove customers that saled very few kinds of products.")
            print("  7. Show current whitelists.")
            print("  8. Clear current whitelist of products.")
            print("  9. Clear current whitelist of customers.")
            opt = 0
            while True:
                o = rc_highlight_in("Now give me your choice (1..9, 0 to exit filtering): ")[0]
                if o.isdigit():
                    opt = int(o)
                    break
            # end while
            if opt == 0:
                break
            rc_warn("REMEMBER: Whitelists are temporary, they will BE CLEARED after you leave filtering and they will NEVER BE SAVED!!!")
            if opt == 1 or opt == 2:
                ids = rc_input("Enter the ids to toggle whitelist states, separating by <spaces>: ").split()
                if opt == 1:
                    for i in ids:
                        si = "0" * (8 - len(i[:8])) + i[:8]
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
                        si = "0" * (9 - len(i[:9])) + i[:9]
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
                        si = "0" * (8 - len(i[:8])) + i[:8]
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
                        si = "0" * (9 - len(i[:9])) + i[:9]
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
                s = rc_input("Tell me the cutoff value (>= 0): ")
                cutoff = -1
                try:
                    cutoff = int(s)
                except ValueError:
                    pass
                if cutoff <= 0:
                    rc_FAIL("NONSENSE! nothing will be changed!")
                else:
                    # removing
                    d = np.nan_to_num(self._ds) > 0.0
                    if opt == 5:
                        l = np.sum(d, axis=1) > cutoff
                        self._ds = self._ds[l, :]
                        # l = ~l
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
                    rc_result("Totally " + str(np.sum(l)) + " items removed.")
                    self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            elif opt == 7:
                l = len(_p_wl)
                rc_result("Whitelist for products has " + str(l) + " items:")
                for i in _p_wl:
                    print("  " + self._products[i] + "[" + i + "]")
                l = len(_c_wl)
                rc_result("Whitelist for customers has " + str(l) + " items:")
                for i in _c_wl:
                    print("  " + self._customers[i] + "[" + i + "]")
            elif opt == 8:
                _p_wl = []
                rc_result("Whitelist of products cleared.")
            else:
                _c_wl = []
                rc_result("Whitelist of customers cleared.")
        # end of outer while loop
        return

    def browse(self):
        axis, coords, zs = None, None, None
        while True:
            axis = rc_input("Enter 'P' for browsing by products, or 'C' for browsing by customers: ")[0].upper()
            if axis == 'P' or axis == 'C':
                break
        while True:
            coords = rc_input("Enter ids to browse, separating by <spaces>: ").split()
            if len(coords) > 0:
                break
        while True:
            zs = rc_input("Show zeros? (Y/N) ").upper()
            if zs != "" and (zs[0] == 'Y' or zs[0] == 'N'):
                zs = zs[0]
                break

        if axis == "P":
            for pid in coords:
                spid = "0" * (8 - len(pid[:8])) + pid[:8]
                if spid not in self._products:
                    rc_fail("Product id: " + spid + " not exists.")
                    continue
                rc_result("Product id: " + spid + ", name: " + self._products[spid] + " weekly average sales:")
                x = self._axis_p.index(spid)
                dx = self._ds[x, :]
                for i in range(len(dx)):
                    if (dx[i] is np.nan or dx[i] == 0.0) and zs == 'N':
                        continue
                    c = "%9.4f by " % dx[i] + self._customers[self._axis_c[i]] + "[" + self._axis_c[i] + "]"
                    print(c)
                rc_state("This product has been saled by " + str(np.sum(dx > 0.0)) + " different customers")
        else:
            for cid in coords:
                scid = "0" * (9 - len(cid[:9])) + cid[:9]
                if scid not in self._customers:
                    rc_fail("Customer id: " + scid + " not exists.")
                    continue
                rc_result("Customer id: " + scid + ", name: " + self._customers[scid] + " weekly average sales:")
                x = self._axis_c.index(scid)
                dx = self._ds[:, x]
                for i in range(len(dx)):
                    if (dx[i] is np.nan or dx[i] == 0.0) and zs == 'N':
                        continue
                    p = "%9.4f of " % dx[i] + self._products[self._axis_p[i]] + "[" + self._axis_p[i] + "]"
                    print(p)
                rc_state("This customer has saled " + str(np.sum(dx > 0.0)) + " kinds of products")
        return

    def plot(self):
        fig = plt.figure()
        ax = fig.gca(projection="3d")
        X = np.arange(self._ds.shape[0])
        Y = np.arange(self._ds.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = np.nan_to_num(self._ds)[X, Y]
        surf = ax.plot_surface(X, Y, Z)
        plt.show()
        return

    def save(self, fn):
        np.savez("ds_" + fn, Mdata=[self._products, self._customers, self._dates], Ds=self._ds)

        with open("ds_" + fn + ".csv", "w", newline="") as f:
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

        self._name = fn
        return

    def __coords(self, axis, pids):
        pcoord = []
        for i in pids:
            pcoord.append(axis.index(i))
        return pcoord

    def load(self, fn, ds_type):
        """
        Load dataset from csv file
        """
        if ds_type == 0:
            # csv
            f = open(fn + ".csv")
            f_csv = csv.reader(f)
            headers = next(f_csv)
            # 0:, 1:customersid, 2:department, 3:enterprise, 4:principal, 5:productname, 6:productid, 7:bizdate, 8:qty
            dates = set()
            p, c = {}, {}
            r = []
            ds = None
            rc_state("Loading sales records...")
            for rec in f_csv:
                dates.add(rec[7].strip())
                p[rec[6].strip()] = rec[5].strip()
                cn = rec[3].strip()
                if '*' in cn:
                    cn = rec[4].strip()
                c[rec[1].strip()] = cn
                r.append((rec[6].strip(), rec[1].strip(), float(rec[8].strip())))
            # end for
            f.close()
            if len(r) == 0:
                return False

            nP = len(p)
            nc = len(c)
            n = nP * nc
            self._name = fn
            self._products = p
            self._customers = c
            self._axis_p = sorted(p)
            self._axis_c = sorted(c)
            self._dates = dates
            self._ds = np.zeros((nP, nc))
            # self._products_to_predict = []
            px = sorted(p)
            pc = sorted(c)
            rc_state("Generating products-customers sales table...")
            for rec in r:
                self._ds[px.index(rec[0]), pc.index(rec[1])] += rec[2]
            # end for
            self._ds = self._ds / len(dates) * 7
            self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            return True
        else:
            # npz
            npz = np.load("ds_" + fn + ".npz")
            rc_state("Loading from early saved dataset...")
            # np.savez("ds_" + fn, Mdata=[self._products, self._customers, self._dates], Ds=self._ds)
            self._name = fn
            self._products = npz["Mdata"][0]
            self._customers = npz["Mdata"][1]
            self._axis_p = sorted(self._products)
            self._axis_c = sorted(self._customers)
            self._dates = npz["Mdata"][2]
            self._ds = npz["Ds"]
            self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            return True

    def states(self):
        s = self._name + ", "
        s += str(len(self._axis_p)) + " products, "
        s += str(len(self._axis_c)) + " customers, "
        s += str(self._num_data) + " non-zero data, "
        s += "dataset density = " + str(round(self.density(), 4))
        return s

    def density(self):
        return self._num_data / self._ds.shape[0] / self._ds.shape[1]

