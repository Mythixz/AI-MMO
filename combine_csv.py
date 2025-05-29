import pandas as pd
import glob

csv_files = glob.glob("pose_dataset/*.csv")
df = pd.concat([pd.read_csv(f) for f in csv_files])
df.to_csv("combined_dataset.csv", index=False)
print("[SUCCESS] Combined CSV created as combined_dataset.csv")
