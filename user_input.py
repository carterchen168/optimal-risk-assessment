import argparse
import os

# first-pass replacement for User_input_accept_generic.m MATLAB file

class params:
    def __init__(self, accept_path, data_path, anomaly_type, load_func, truth_func):
        self.accept_path = accept_path
        self.data_path = data_path
        self.anomaly_type = anomaly_type
        self.load_func = load_func
        self.truth_func = truth_func

    def show_fields(self):
        print(f"accept_path: {self.accept_path}")
        print(f"data_path: {self.data_path}")
        print(f"anomaly_type: {self.anomaly_type}")
        print(f"load_func: {self.load_func}")
        print(f"truth_func: {self.truth_func}")

def main():
    test = params(os.getenv("ACCEPT_DIR"), os.getenv("DATA_DIR"), ['Adverse Event 1','Adverse Event 2','Adverse Event 3'], ['Function_1','Function_2','Function_3'], ['GTFunc_1','GTFunc_2','GTFunc_3'])
    test.show_fields()

if __name__ == "__main__":
    main()