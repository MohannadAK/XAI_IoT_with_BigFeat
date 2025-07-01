import pandas as pd
import os

def analyze_device_files():
    """
    Process device1 to device9 CSV files and create two summary CSV files:
    1. F1 scores summary with devices as rows and specific values as columns
    2. Training times summary with devices as rows and specific values as columns
    """

    # Initialize dictionaries to store data for each device
    f1_data = {}
    time_data = {}

    # Process files from device1 to device9
    for device_num in range(1, 10):
        filename = f"../Enhanced_Ensemble_TPOT_ExtraTrees_Results/device{device_num}_comprehensive_comparison.csv"

        try:
            # Read the CSV file
            df = pd.read_csv(filename)
            print(f"Processing {filename}...")

            # Debug: Print file existence check
            if not os.path.exists(filename):
                print(f"DEBUG: File does not exist: {filename}")
                continue

            # Extract F1 scores
            # Original: F1 score from ExtraTrees row, Original F1 Score column
            original_f1 = df[df['Model'] == 'ExtraTrees']['Original F1 Score'].iloc[0] if len(df[df['Model'] == 'ExtraTrees']) > 0 else None

            # BigFeat: F1 score from ExtraTrees row, BigFeat F1 Score column
            bigfeat_f1 = df[df['Model'] == 'ExtraTrees']['BigFeat F1 Score'].iloc[0] if len(df[df['Model'] == 'ExtraTrees']) > 0 else None

            # TPOT: F1 score from TPOT (ExtraTrees) row, Original F1 Score column
            tpot_f1 = df[df['Model'] == 'TPOT (ExtraTrees)']['Original F1 Score'].iloc[0] if len(df[df['Model'] == 'TPOT (ExtraTrees)']) > 0 else None

            # BigFeat + TPOT: F1 score from TPOT (ExtraTrees) row, BigFeat F1 Score column
            bigfeat_tpot_f1 = df[df['Model'] == 'TPOT (ExtraTrees)']['BigFeat F1 Score'].iloc[0] if len(df[df['Model'] == 'TPOT (ExtraTrees)']) > 0 else None

            # Store F1 data
            f1_data[f'device{device_num}'] = {
                'Original': original_f1,
                'BigFeat': bigfeat_f1,
                'TPOT': tpot_f1,
                'BigFeat + TPOT': bigfeat_tpot_f1
            }

            # Extract training times
            # Original: Training time from ExtraTrees row, Original Training Time column
            original_time = df[df['Model'] == 'ExtraTrees']['Original Training Time'].iloc[0] if len(df[df['Model'] == 'ExtraTrees']) > 0 else None

            # BigFeat: Training time from ExtraTrees row, BigFeat Training Time column
            bigfeat_time = df[df['Model'] == 'ExtraTrees']['BigFeat Training Time'].iloc[0] if len(df[df['Model'] == 'ExtraTrees']) > 0 else None

            # TPOT: Training time from TPOT (ExtraTrees) row, Original Training Time column
            tpot_time = df[df['Model'] == 'TPOT (ExtraTrees)']['Original Training Time'].iloc[0] if len(df[df['Model'] == 'TPOT (ExtraTrees)']) > 0 else None

            # BigFeat + TPOT: Training time from TPOT (ExtraTrees) row, BigFeat Training Time column
            bigfeat_tpot_time = df[df['Model'] == 'TPOT (ExtraTrees)']['BigFeat Training Time'].iloc[0] if len(df[df['Model'] == 'TPOT (ExtraTrees)']) > 0 else None

            # Store time data
            time_data[f'device{device_num}'] = {
                'Original': original_time,
                'BigFeat': bigfeat_time,
                'TPOT': tpot_time,
                'BigFeat + TPOT': bigfeat_tpot_time
            }

            print(f"✓ Successfully processed {filename}")

        except FileNotFoundError:
            print(f"⚠ Warning: {filename} not found, skipping...")
            continue
        except Exception as e:
            print(f"✗ Error processing {filename}: {str(e)}")
            continue

    # Create F1 scores DataFrame
    f1_df = pd.DataFrame.from_dict(f1_data, orient='index')
    f1_df.index.name = 'Device'

    # Create training times DataFrame
    time_df = pd.DataFrame.from_dict(time_data, orient='index')
    time_df.index.name = 'Device'

    # Define output file names
    f1_output_file = 'f1_scores_summary.csv'
    time_output_file = 'training_times_summary.csv'

    # Save to CSV files
    f1_df.to_csv(f1_output_file)
    time_df.to_csv(time_output_file)

    print(f"\n{'=' * 60}")
    print(f"✓ Output files created successfully:")
    print(f"  1. F1 Scores Summary: {f1_output_file}")
    print(f"  2. Training Times Summary: {time_output_file}")
    print(f"{'=' * 60}")

    # Display preview of the results
    print(f"\n📊 F1 Scores Summary Preview:")
    print(f1_df.round(4))

    print(f"\n⏱ Training Times Summary Preview:")
    print(time_df.round(2))

    return f1_df, time_df

def validate_data_structure(
        sample_file="../Enhanced_Ensemble_TPOT_ExtraTrees_Results/device1_comprehensive_comparison.csv"):
    """
    Validate the expected data structure of the CSV files
    """
    try:
        df = pd.read_csv(sample_file)
        print(f"📋 Data Structure Validation for {sample_file}:")
        print(f"Columns: {list(df.columns)}")
        print(f"Shape: {df.shape}")
        print(f"Models: {list(df['Model'])}")
        print(f"TPOT models: {list(df[df['Model'].str.contains('TPOT', case=False, na=False)]['Model'])}")
        return True
    except FileNotFoundError:
        print(f"⚠ Sample file {sample_file} not found for validation")
        return False
    except Exception as e:
        print(f"✗ Error validating data structure: {str(e)}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Device Analysis Script")
    print("=" * 60)

    # Validate data structure first
    if validate_data_structure():
        print("\n" + "=" * 60)

        # Run main processing
        f1_results, time_results = analyze_device_files()

        print(f"\n🎉 Analysis complete!")
    else:
        print("❌ Data validation failed. Please check your CSV files.")