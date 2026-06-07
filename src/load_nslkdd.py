"""
NSL-KDD loader for CNN-LSTM-DQN-IDS-v2.

NSL-KDD ships as headerless KDDTrain+.txt / KDDTest+.txt. This adds the 41
column names, maps each fine attack name to its coarse category
{normal,dos,probe,r2l,u2r}, writes a 'category' column, and saves a CSV that
preprocess.py --dataset nslkdd can consume directly.

Usage:
    python src/load_nslkdd.py --input data/nslkdd/KDDTrain+.txt \
        --out data/nslkdd/train_with_category.csv
    python src/preprocess.py --dataset nslkdd \
        --input data/nslkdd/train_with_category.csv --out data/processed/nslkdd_train
"""

import argparse
import pandas as pd

COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Fine attack -> coarse category (standard NSL-KDD grouping)
ATTACK_CATEGORY = {
    "normal": "normal",
    # DoS
    "neptune": "dos", "back": "dos", "land": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "mailbomb": "dos",
    "apache2": "dos", "processtable": "dos", "udpstorm": "dos", "worm": "dos",
    # Probe
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "sendmail": "r2l", "named": "r2l", "snmpgetattack": "r2l",
    "snmpguess": "r2l", "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}


def build(input_path, out_path):
    df = pd.read_csv(input_path, header=None, names=COLUMNS)
    df["category"] = df["label"].astype(str).str.strip().map(ATTACK_CATEGORY)
    unmapped = df["category"].isna().sum()
    if unmapped:
        print(f"WARNING {unmapped} rows had unmapped attack labels -> set to 'r2l'."
              f" Unique: {sorted(df.loc[df['category'].isna(),'label'].unique())}")
        df["category"] = df["category"].fillna("r2l")
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path}  ({len(df)} rows)")
    print(df["category"].value_counts().to_dict())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.input, args.out)


if __name__ == "__main__":
    main()
