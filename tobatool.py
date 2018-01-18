
import csv

raw_data = []

while True:
    print("0. Initialize database")
    print("1. Load new data from .csv file")
    print("2. Combine data with history")
    print("3. Save current data")
    cmd = input("Please choice task ('q' to exit): ").strip()
    if cmd == 'q':
        break
    elif cmd == '1':
        if len(raw_data) > 0:
            print("WARN: You have new data not combined yet. If you read more new data, these data will lost.")
            cmd = input("Enter (Y) to read or anything else to stop: ").strip().capitalize()
            if cmd != 'Y':
                continue
        fn = input("Please enter the filename: ").strip()
        load_data(fn)
    else:
        pass


def load_data(fn):
    print("Loading...")

    print("Totally " + str(n) + " lines of new data loaded.")
    return
