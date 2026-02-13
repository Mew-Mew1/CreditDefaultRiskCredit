import pandas as pd
import os

def convert_xls_to_csv(
    input_path="data/raw/default_of_credit_card_clients.xls",
    output_path="data/raw_csv/default_of_credit_card_clients.csv"
):
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load Excel file (skip first row which is metadata)
    df = pd.read_excel(input_path, header=1)

    # Save as CSV
    df.to_csv(output_path, index=False)
    print(f"Converted {input_path} → {output_path}")

if __name__ == "__main__":
    convert_xls_to_csv()