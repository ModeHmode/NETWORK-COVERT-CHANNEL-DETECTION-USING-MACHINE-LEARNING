# Imports
import argparse
from lib import utilities as utils
from joblib import dump
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier


# Define expected arguments
parser = argparse.ArgumentParser(
    prog="create_model",
    description="Creates a model for AI detection of covert channels",
)
parser.add_argument(
    "-cf",
    "--covert-file",
    "--covert-files",
    nargs="+",
    required=True,
    help="pcap file(s) containing covert packets",
)
parser.add_argument(
    "-nf",
    "--non-covert-file",
    "--non-covert-files",
    nargs="+",
    required=True,
    help="pcap file(s) containing non-covert packets",
)
parser.add_argument(
    "-cn",
    "--covert-count",
    type=int,
    default=-1,
    help="number of covert packets to read per file, default is -1 (unlimited)",
)
parser.add_argument(
    "-cm",
    "--covert-max",
    type=int,
    default=-1,
    help="max number of covert packets to read, default is -1 (unlimited)",
)
parser.add_argument(
    "-nn",
    "--non-covert-count",
    type=int,
    default=-1,
    help="number of non-covert packets to read per file, default is -1 (unlimited)",
)
parser.add_argument(
    "-nm",
    "--non-covert-max",
    type=int,
    default=-1,
    help="max number of non-covert packets to read, default is -1 (unlimited)",
)
parser.add_argument(
    "-o",
    "--output",
    required=True,
    help="filename of output model",
)
parser.add_argument(
    "-v",
    "--verbosity",
    action="count",
    default=0,
    help="increase output verbosity",
)
algo_group = parser.add_mutually_exclusive_group()
algo_group.add_argument(
    "-dtr",
    "--decision-tree",
    action="store_const",
    dest="algorithm",
    const="dtr",
    default="dtr",
    help="use Decision Tree as model algorithm",
)
algo_group.add_argument(
    "-knn",
    "--k-nearest-neighbors",
    action="store_const",
    dest="algorithm",
    const="knn",
    help="use K Nearest Neighbors as model algorithm",
)
algo_group.add_argument(
    "-lrg",
    "--logistic-regression",
    action="store_const",
    dest="algorithm",
    const="lrg",
    help="use Logistic Regression as model algorithm",
)
algo_group.add_argument(
    "-gnb",
    "--naive-bayes",
    action="store_const",
    dest="algorithm",
    const="gnb",
    help="use Gaussian Naive Bayes as model algorithm",
)
algo_group.add_argument(
    "-svm",
    "--support-vector",
    action="store_const",
    dest="algorithm",
    const="svm",
    help="use Support Vector Machine as model algorithm",
)
algo_group.add_argument(
    "-mlp",
    "--multi-layer_perceptron",
    action="store_const",
    dest="algorithm",
    const="mlp",
    help="use Multi-Layer Perceptron as model algorithm",
)


# Parse the arguments received
args = parser.parse_args()
if args.verbosity >= 3:
    print("Arguments namespace: ", end="")
    print(args)


# Check if pcap files exists and is/are readable
if args.verbosity >= 1:
    print("Checking file access")

if utils.check_file_readable(args.covert_file, args.non_covert_file) > 0:
    if args.verbosity >= 1:
        print("Program cannot proceed, please check pcap file(s) specified.")
        print("Exiting...")
    exit()


# Read in covert pcap data
if args.verbosity >= 1:
    print("Reading in covert packets")
covert_packets = utils.read_pcap_files(
    args.covert_file,
    maximum_count=args.covert_max,
    file_count=args.covert_count,
    verbosity=args.verbosity,
)
if args.verbosity >= 1:
    print("Total covert packets read: {}".format(len(covert_packets)))


# Read in non-covert pcap data
if args.verbosity >= 1:
    print("Reading in non-covert packets")
non_covert_packets = utils.read_pcap_files(
    args.non_covert_file,
    maximum_count=args.non_covert_max,
    file_count=args.non_covert_count,
    verbosity=args.verbosity,
)
if args.verbosity >= 1:
    print("Total non-covert packets read: {}".format(len(non_covert_packets)))


# Perform feature engineering
if args.verbosity >= 1:
    print("Feature engineering packets")

if args.verbosity >= 2:
    print("Feature engineering covert packets")
covert_df = utils.packets_to_dataframe(covert_packets, 1)
if args.verbosity >= 3:
    print("covert dataframe:")
    print(covert_df)

if args.verbosity >= 2:
    print("Feature engineering non-covert packets")
non_covert_df = utils.packets_to_dataframe(non_covert_packets, 0)
if args.verbosity >= 3:
    print("non-covert dataframe:")
    print(non_covert_df)


# Final preparations before training
# Merge, randomize order, extract labels, and scale
if args.verbosity >= 1:
    print("Prepare data before training")

if args.verbosity >= 2:
    print("Merge and randomize packets")
train = [covert_df, non_covert_df]
train = utils.randomize_dataframe(train)

if args.verbosity >= 2:
    print("Extract label")
labels = train["is_covert"]
train.drop(labels=["is_covert"], axis=1, inplace=True)

if args.verbosity >= 2:
    print("Scale data")
scaler = StandardScaler()
train = scaler.fit_transform(train)

if args.verbosity >= 3:
    print("Final training data")
    print(train)
    print("Final training data length: ", end="")
    print(len(train))
    print("Final labels")
    print(labels)


# Choose the model
model = None
if args.verbosity >= 1:
    print("Preparing model")

if args.algorithm == 'knn':
    model = KNeighborsClassifier(n_neighbors=2)
elif args.algorithm == 'gnb':
    model = GaussianNB(var_smoothing=0.0012328467394420659)
elif args.algorithm == 'lrg':
    model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
elif args.algorithm == 'dtr':
    model = DecisionTreeClassifier(criterion='gini', max_depth=None, splitter='best')
elif args.algorithm == 'svm':
    model = LinearSVC(penalty='l1', loss='squared_hinge', dual=False, C=100.0)
elif args.algorithm == 'mlp':
    model = MLPClassifier(hidden_layer_sizes=(16, 8, 4), activation="relu")


# Train the model
if args.verbosity >= 1:
    print("Training model")

model.fit(train, labels)


# Save the model to file
if args.verbosity >= 1:
    print("Saving model file")
dump((model, scaler), args.output)


# Done
print("Creating model finished")