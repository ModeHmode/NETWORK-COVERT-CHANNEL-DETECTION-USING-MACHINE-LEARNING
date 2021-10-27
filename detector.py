import argparse
import scapy.all as scapy
import pandas as pd
from joblib import load
from lib import utilities as utils
from lib.feature_engineering import extract_feature


# Define expected arguments
parser = argparse.ArgumentParser(
    prog="create_model",
    description="Creates a model for AI detection of covert channels",
)
parser.add_argument(
    "model",
    help="model file to use to detect covert packets",
)
parser.add_argument(
    "-i",
    "--interface",
    help="interface to use when monitoring packets",
)
parser.add_argument(
    "-v",
    "--verbosity",
    action="count",
    default=0,
    help="increase output verbosity",
)


# Parse the arguments received
args = parser.parse_args()
if args.verbosity >= 3:
    print("Arguments namespace: ", end="")
    print(args)


# Check if model exists and is/are readable
if args.verbosity >= 1:
    print("Checking file access")

if utils.check_file_readable([args.model]) > 0:
    if args.verbosity >= 1:
        print("Program cannot proceed, please model file specified.")
        print("Exiting...")
    exit()


# Check if interface passed in exists
if args.interface:
    if args.verbosity >= 1:
        print("Checking interface")

    if not args.interface in scapy.get_if_list():
        if args.verbosity >= 1:
            print("Program cannot proceed, cannot find interface specified.")
            print("Exiting...")
        exit()


# Read in covert pcap data
if args.verbosity >= 1:
    print("Loading model from file")

model, scaler = load(args.model)
if args.verbosity >= 3:
    print("model:")
    print(model)
    print("scaler:")
    print(scaler)


# Function to check packet if contains covert data
def detect_covert(packet):
    if args.verbosity >= 3:
        print(packet.summary())

    features = extract_feature(packet)
    packets = [features]
    test = pd.DataFrame(packets)
    test = scaler.transform(test)
    result = model.predict(test)
    if result[0]:
        print("ALERT: Potential covert packet found!")


if args.interface:
    print("Started monitoring packets from interface {}".format(args.interface))
    scapy.sniff(prn=detect_covert, iface=args.interface)

else:
    print("Started monitoring packets from all interfaces")
    scapy.sniff(prn=detect_covert)
