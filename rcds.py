# -*- coding: GBK -*-
import numpy as np
import csv
import os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class Dataset:
    """Dataset"""
    _name = None        # raw data file, csv, sales records
    _products = {}              # products in the dataset
    _customers = {}             # customers in the dataset
    _axis_p, _axis_c = [], []   # axis of products and customers
    _dates = set()              # dates in the dataset
    _num_data = 0               # number of non-zero data in the dataset
    _ds = None                  # dataset main table
    # _products_to_predict = []   # ids of products to predict

    def __init__(self):
        return

    def filter(self):
        _p_wl = []                  # whitelist of products
        _c_wl = []                  # whitelist of customers
        while True:
            print("You have six options to filter the dataset:")
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
                o = input("Now give me your choice (1..9, 0 to exit filtering): ").strip()[0]
                if o.isdigit():
                    opt = int(o)
                    break
            if opt == 0:
                break
            print("REMEMBER: Whitelists are temporary, they will BE CLEARED after you leave filtering and they will NEVER BE SAVED!!!")
            if opt == 1 or opt == 2:
                ids = input("Enter the ids to toggle whitelist states, separating by <spaces>: ").strip().split()
                if opt == 1:
                    for i in ids:
                        si = "0" * (8 - len(i[:8])) + i[:8]
                        if si not in self._products:
                            print("Product id:", si, ", not exists.")
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
                            print("Customer id:", si, ", not exists.")
                            continue
                        if si in _c_wl:
                            _c_wl.remove(si)
                            print("Customer id:", si, ", name:", self._customers[si], "is removed from whitelist.")
                        else:
                            _c_wl.append(si)
                            print("Customer id:", si, ", name:", self._customers[si], "is put into whitelist.")
            elif opt == 3 or opt == 4:
                ids = input("Enter the ids to remove from dataset, separating by <spaces>: ").strip().split()
                if opt == 3:
                    for i in ids:
                        si = "0" * (8 - len(i[:8])) + i[:8]
                        if si not in self._products:
                            print("Product id:", si, "not exists.")
                            continue
                        if si in _p_wl:
                            print("Product id:", si, "is in the whitelist, cannot remove it.")
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
                            print("Customer id:", si, " not exists.")
                            continue
                        if si in _c_wl:
                            print("Customer id:", si, "is in the whitelist, cannot remove it.")
                            continue
                        coord = self._axis_c.index(si)
                        name = self._customers.pop(si)
                        self._axis_c.remove(si)
                        self._ds = np.delete(self._ds, coord, axis=1)
                        print("Customer id:", si, ", name:", name, "is removed from dataset.")
                self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            elif opt == 5 or opt == 6:
                s = input("Tell me the cutoff value (>= 0): ").strip()
                cutoff = -1
                try:
                    cutoff = int(s)
                except ValueError:
                    pass
                if cutoff <= 0:
                    print("NONSENSE! nothing will be changed!")
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
                    print("Totally", np.sum(l), "items removed.")
                    self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            elif opt == 7:
                l = len(_p_wl)
                print("Whitelist for products has", l, "items:")
                for i in _p_wl:
                    print("  " + self._products[i] + "[" + i + "]")
                l = len(_c_wl)
                print("Whitelist for customers has", l, "items:")
                for i in _c_wl:
                    print("  " + self._customers[i] + "[" + i + "]")
            elif opt == 8:
                _p_wl = []
            else:
                _c_wl = []
        # end of outer while loop
        return

    def browse(self):
        axis, coords, zs = None, None, None
        while True:
            axis = input("Enter 'P' for browsing by products, or 'C' for browsing by customers: ").strip()[0].upper()
            if axis == 'P' or axis == 'C':
                break
        while True:
            coords = input("Enter ids to browse, separating by <spaces>: ").strip().split()
            if len(coords) > 0:
                break
        while True:
            zs = input("Show zeros? (Y/N) ").strip()[0].upper()
            if (zs == 'Y' or zs == 'N'):
                break

        if axis == "P":
            for pid in coords:
                spid = "0" * (8 - len(pid[:8])) + pid[:8]
                if spid not in self._products:
                    print("Product id:", spid, ", not exists.")
                    continue
                print("Product id:", spid, ", name:", self._products[spid], "weekly average sales:")
                x = self._axis_p.index(spid)
                dx = self._ds[x, :]
                for i in range(len(dx)):
                    if (dx[i] is np.nan or dx[i] == 0.0) and zs == 'N':
                        continue
                    c = "%9.4f by " % dx[i] + self._customers[self._axis_c[i]] + "[" + self._axis_c[i] + "]"
                    print(c)
                print("This product has been saled by", np.sum(dx > 0.0), "different customers")
        else:
            for cid in coords:
                scid = "0" * (9 - len(cid[:9])) + cid[:9]
                if scid not in self._customers:
                    print("Customer id:", scid, ", not exists.")
                    continue
                print("Customer id:", scid, ", name:", self._customers[scid] + " weekly average sales:")
                x = self._axis_c.index(scid)
                dx = self._ds[:, x]
                for i in range(len(dx)):
                    if (dx[i] is np.nan or dx[i] == 0.0) and zs == 'N':
                        continue
                    p = "%9.4f of " % dx[i] + self._products[self._axis_p[i]] + "[" + self._axis_p[i] + "]"
                    print(p)
                print("This customer has saled", np.sum(dx > 0.0), "kinds of products")
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
        np.savez(fn, Mdata=[self._products, self._customers, self._dates], Ds=self._ds)

        with open(fn + ".csv", "w", newline="") as f:
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
            print("Loading sales records...", flush=True)
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
            print("Generating products-customers sales table...", flush=True)
            for rec in r:
                self._ds[px.index(rec[0]), pc.index(rec[1])] += rec[2]
            # end for
            self._ds = self._ds / len(dates) * 7
            self._num_data = np.sum(np.nan_to_num(self._ds) > 0.0)
            return True
        else:
            # npz
            npz = np.load(fn + ".npz")
            print("Loading from early saved dataset...", flush=True)
            # np.savez(fn, Prod=self._products, Cust=self._customers, Dates=self._dates, Ds=self._ds)
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

