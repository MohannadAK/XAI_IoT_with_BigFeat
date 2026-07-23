import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

def recreate_heatmaps():
    results_dir = '/home/victoryst/personal/dev/XAI_IoT_with_BigFeat/Results/IoT/Feature_Importance/SHAP_Feature_Importance_Analysis'
    csv_files = glob.glob(os.path.join(results_dir, '*_shap_feature_importance_comparison.csv'))
    
    if not csv_files:
        print("No CSV files found in", results_dir)
        return
        
    for csv_file in csv_files:
        device_name = os.path.basename(csv_file).split('_')[0]
        print(f"Processing {device_name}...")
        
        df = pd.read_csv(csv_file)
        
        # Pivot the dataframe
        # We want Models on Y axis, Features on X axis
        pivot_df = df.pivot(index='Model', columns='Feature', values='SHAP_Value')
        
        # Fill missing values with 0
        pivot_df = pivot_df.fillna(0)
        
        # Ensure we have a specific order for models on Y axis if possible
        expected_models = ['ExtraTrees Original', 'ExtraTrees BigFeat', 'TPOT Original', 'TPOT BigFeat']
        actual_models = [m for m in expected_models if m in pivot_df.index]
        for m in pivot_df.index:
            if m not in actual_models:
                actual_models.append(m)
        pivot_df = pivot_df.reindex(actual_models)
        
        # Sort features by average importance across all models
        avg_importance = pivot_df.mean()
        sorted_features = avg_importance.sort_values(ascending=False).index
        pivot_df = pivot_df[sorted_features]
        
        # Plotting
        plt.figure(figsize=(22, 12))
        ax = sns.heatmap(pivot_df, annot=True, fmt='.4f', cmap='viridis', 
                         annot_kws={"size": 12},
                         cbar_kws={'label': 'SHAP Value'})
        
        # Increase colorbar font sizes
        cbar = ax.collections[0].colorbar
        cbar.ax.yaxis.label.set_size(18)
        cbar.ax.tick_params(labelsize=14)
        
        plt.title(f'{device_name} - SHAP Feature Importance Comparison (All Features)', fontsize=24, pad=20, fontweight='bold')
        plt.xlabel('Features (Sorted by Average Importance)', fontsize=20, fontweight='bold', labelpad=15)
        plt.ylabel('Models', fontsize=20, fontweight='bold', labelpad=15)
        
        # Rotate x-axis labels and color them based on feature type
        plt.xticks(rotation=45, ha='right', fontsize=14)
        plt.yticks(rotation=45, fontsize=16)
        
        # Color the x-tick labels: Green for original, Blue for BigFeat
        for tick_label in ax.get_xticklabels():
            feature_name = tick_label.get_text()
            if 'BigFeat' in feature_name:
                tick_label.set_color('#6A0DAD') # Purple/Blue for BigFeat to match image or let's use blue/purple. In the image BigFeat is purple. 
            else:
                tick_label.set_color('#006400') # Dark green for original
                
        # Make the tick labels bold
        for tick_label in ax.get_xticklabels():
            tick_label.set_fontweight('bold')
        for tick_label in ax.get_yticklabels():
            tick_label.set_fontweight('bold')
            
        plt.tight_layout()
        
        # Save as jpg and pdf
        output_jpg = os.path.join(results_dir, f"{device_name}_all_features_heatmap.jpg")
        output_pdf = os.path.join(results_dir, f"{device_name}_all_features_heatmap.pdf")
        
        plt.savefig(output_jpg, dpi=300, bbox_inches='tight')
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(output_pdf) as pdf:
            pdf.savefig(plt.gcf(), bbox_inches='tight')
            
        plt.close()
        print(f"Saved heatmaps for {device_name}")

if __name__ == '__main__':
    recreate_heatmaps()
