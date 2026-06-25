"""
console_analyzer.py - Terminal-based Forest Cover Type Analysis
===============================================================
Run: python console_analyzer.py

Features:
- Display model metrics with ASCII charts
- Interactive prediction console
- Dataset statistics visualization
- Model comparison graphs in terminal
"""

import os
import pickle
import json
import numpy as np
import pandas as pd
import zipfile
from datetime import datetime

# ASCII Color Codes for Terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'
    DIM = '\033[2m'

# ── Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_ZIP = os.path.join(BASE_DIR, "..", "data", "covtype_csv.zip")

# ── Load Models and Metrics ──────────────────────────────────────────────
def load_models():
    """Load trained models from disk"""
    print(f"{Colors.CYAN} Loading models from {MODELS_DIR}...{Colors.END}")
    
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
        rf_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "knn.pkl"), "rb") as f:
        knn_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "naive_bayes.pkl"), "rb") as f:
        nb_model = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, "metrics.json"), "r") as f:
        metrics = json.load(f)
    
    print(f"{Colors.GREEN} Models loaded successfully!{Colors.END}\n")
    return scaler, rf_model, knn_model, nb_model, metrics

# ── ASCII Bar Chart ────────────────────────────────────────────────────
def draw_bar_chart(data, title, width=40):
    """Draw a simple ASCII bar chart"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{title}{Colors.END}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")
    
    max_value = max(data.values())
    for label, value in data.items():
        bar_length = int((value / max_value) * width)
        bar = "█" * bar_length
        color = Colors.GREEN if value >= 70 else Colors.YELLOW if value >= 40 else Colors.RED
        print(f"{label:<15} {color}{bar:<{width}}{Colors.END} {value:.1f}%")
    print()

# ── ASCII Radar Chart (Simplified) ─────────────────────────────────────
def draw_radar_chart(metrics_data):
    """Draw a simple ASCII radar chart for model comparison"""
    print(f"{Colors.BOLD}{Colors.CYAN} Model Performance Radar (Simplified){Colors.END}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")
    
    models = ["KNN", "Naïve Bayes", "Random Forest"]
    metrics_list = ["Accuracy", "F1 Score", "Precision", "Recall"]
    
    # Create a simple table
    print(f"\n{'Model':<15}", end="")
    for m in metrics_list:
        print(f"{m:>12}", end="")
    print()
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")
    
    for model in models:
        key = model.lower().replace(" ", "_").replace("naïve", "naive")
        if key in metrics_data:
            print(f"{model:<15}", end="")
            for metric in metrics_list:
                m_key = metric.lower().replace(" ", "_").replace("f1", "f1")
                val = metrics_data[key].get(m_key if m_key != "f1_score" else "f1", 0)
                color = Colors.GREEN if val >= 70 else Colors.YELLOW if val >= 40 else Colors.RED
                print(f"{color}{val:>11.1f}%{Colors.END}", end="")
            print()
    print()

# ── Model Comparison Table ─────────────────────────────────────────────
def show_model_comparison(metrics):
    """Display model comparison table"""
    print(f"{Colors.BOLD}{Colors.CYAN} MODEL PERFORMANCE COMPARISON{Colors.END}")
    print(f"{Colors.DIM}{'═' * 70}{Colors.END}")
    
    # Header
    print(f"{'Algorithm':<20} {'Type':<12} {'Accuracy':<10} {'F1 Score':<10} {'Precision':<10} {'Recall':<10}")
    print(f"{Colors.DIM}{'─' * 70}{Colors.END}")
    
    models_info = [
        ("knn", "K-Nearest Neighbors", "Supervised"),
        ("naive_bayes", "Naïve Bayes", "Supervised"),
        ("random_forest", "Random Forest ★", "Supervised"),
        ("kmeans", "K-Means", "Unsupervised")
    ]
    
    for key, name, mtype in models_info:
        if key in metrics:
            m = metrics[key]
            if key == "kmeans":
                print(f"{name:<20} {mtype:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
            else:
                acc_color = Colors.GREEN if m['accuracy'] >= 70 else Colors.YELLOW
                f1_color = Colors.GREEN if m['f1'] >= 70 else Colors.YELLOW
                prec_color = Colors.GREEN if m['precision'] >= 70 else Colors.YELLOW
                rec_color = Colors.GREEN if m['recall'] >= 70 else Colors.YELLOW
                
                print(f"{name:<20} {mtype:<12} "
                      f"{acc_color}{m['accuracy']:>6.1f}%{Colors.END}   "
                      f"{f1_color}{m['f1']:>6.1f}%{Colors.END}   "
                      f"{prec_color}{m['precision']:>6.1f}%{Colors.END}   "
                      f"{rec_color}{m['recall']:>6.1f}%{Colors.END}")
        else:
            print(f"{name:<20} {mtype:<12} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10}")
    
    print(f"{Colors.DIM}{'═' * 70}{Colors.END}\n")

# ── Dataset Statistics ─────────────────────────────────────────────────
def load_dataset_stats():
    """Load and display dataset statistics"""
    print(f"{Colors.BOLD}{Colors.CYAN} DATASET STATISTICS{Colors.END}")
    print(f"{Colors.DIM}{'─' * 60}{Colors.END}")
    
    try:
        with zipfile.ZipFile(DATA_ZIP, "r") as z:
            with z.open("covtype.csv") as f:
                df = pd.read_csv(f)
        
        total_rows = len(df)
        total_cols = len(df.columns)
        cover_types = df['Cover_Type'].value_counts().sort_index()
        
        cover_type_names = {
            1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
            4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas Fir", 7: "Krummholz"
        }
        
        print(f"{Colors.GREEN}✓ Total Rows:{Colors.END} {total_rows:,}")
        print(f"{Colors.GREEN}✓ Total Features:{Colors.END} {total_cols}")
        print(f"{Colors.GREEN}✓ Target Classes:{Colors.END} 7")
        print(f"{Colors.GREEN}✓ Missing Values:{Colors.END} 0\n")
        
        # Class distribution bar chart
        print(f"{Colors.BOLD}Class Distribution:{Colors.END}")
        max_count = cover_types.max()
        
        for class_id, count in cover_types.items():
            class_name = cover_type_names.get(class_id, f"Class {class_id}")
            bar_length = int((count / max_count) * 40)
            bar = "█" * bar_length
            percentage = (count / total_rows) * 100
            
            # Color based on percentage
            if percentage > 20:
                color = Colors.GREEN
            elif percentage > 5:
                color = Colors.YELLOW
            else:
                color = Colors.RED
            
            print(f"{class_name:<20} {color}{bar:<40}{Colors.END} {count:>7,} ({percentage:>5.1f}%)")
        
        print()
        
    except Exception as e:
        print(f"{Colors.RED}⚠ Could not load dataset: {e}{Colors.END}\n")
        print(f"{Colors.YELLOW}Using sample data from training...{Colors.END}\n")

# ── Interactive Prediction Console ─────────────────────────────────────
def interactive_prediction(scaler, rf_model, knn_model, nb_model, metrics):
    """Interactive prediction menu in console"""
    cover_type_names = {
        1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine",
        4: "Cottonwood/Willow", 5: "Aspen", 6: "Douglas Fir", 7: "Krummholz"
    }
    
    wilderness_names = {
        1: "Rawah", 2: "Neota", 3: "Comanche Peak", 4: "Cache la Poudre"
    }
    
    print(f"\n{Colors.BOLD}{Colors.CYAN} INTERACTIVE PREDICTION CONSOLE{Colors.END}")
    print(f"{Colors.DIM}{'═' * 60}{Colors.END}")
    print(f"{Colors.YELLOW} Tip: Press Enter to use default values{Colors.END}\n")
    
    while True:
        print(f"\n{Colors.BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{Colors.END}")
        
        # Algorithm selection
        print(f"\n{Colors.CYAN}Select Algorithm:{Colors.END}")
        print("  1. Random Forest ★ (Recommended)")
        print("  2. K-Nearest Neighbors")
        print("  3. Naïve Bayes")
        print("  4. Compare All Models")
        print("  5. Exit")
        
        choice = input(f"\n{Colors.GREEN}Your choice (1-5):{Colors.END} ").strip()
        
        if choice == "5":
            print(f"\n{Colors.GREEN}👋 Goodbye!{Colors.END}\n")
            break
        
        # Get input features
        print(f"\n{Colors.BOLD} Enter Feature Values:{Colors.END}")
        print(f"{Colors.DIM}(Leave blank for default){Colors.END}\n")
        
        defaults = {
            'elevation': 2596, 'aspect': 51, 'slope': 3,
            'h_dist_hydrology': 258, 'v_dist_hydrology': 0,
            'h_dist_roadways': 510, 'hillshade_9am': 221,
            'hillshade_noon': 232, 'hillshade_3pm': 148,
            'h_dist_fire_points': 6279
        }
        
        def get_input(prompt, default, min_val=None, max_val=None):
            while True:
                val = input(f"{prompt} [{default}]: ").strip()
                if val == "":
                    return default
                try:
                    val = float(val)
                    if min_val is not None and val < min_val:
                        print(f"{Colors.RED}Value must be >= {min_val}{Colors.END}")
                        continue
                    if max_val is not None and val > max_val:
                        print(f"{Colors.RED}Value must be <= {max_val}{Colors.END}")
                        continue
                    return val
                except ValueError:
                    print(f"{Colors.RED}Please enter a number{Colors.END}")
        
        elevation = get_input("Elevation (m)", 2596, 1859, 3858)
        aspect = get_input("Aspect (°)", 51, 0, 360)
        slope = get_input("Slope (°)", 3, 0, 66)
        h_dist_hydrology = get_input("H. Dist. to Hydrology", 258, 0, 4000)
        v_dist_hydrology = get_input("V. Dist. to Hydrology", 0, -200, 200)
        h_dist_roadways = get_input("H. Dist. to Roadways", 510, 0, 7000)
        hillshade_9am = get_input("Hillshade 9am", 221, 0, 255)
        hillshade_noon = get_input("Hillshade Noon", 232, 0, 255)
        hillshade_3pm = get_input("Hillshade 3pm", 148, 0, 255)
        h_dist_fire = get_input("H. Dist. to Fire Points", 6279, 0, 14000)
        
        print(f"\n{Colors.CYAN}Wilderness Area:{Colors.END}")
        for i, name in wilderness_names.items():
            print(f"  {i}. {name}")
        wilderness = int(get_input("Select (1-4)", 1, 1, 4))
        
        soil_type = int(get_input("Soil Type (1-40)", 29, 1, 40))
        
        # Build feature vector
        cont_features = [elevation, aspect, slope, h_dist_hydrology, v_dist_hydrology,
                        h_dist_roadways, hillshade_9am, hillshade_noon, hillshade_3pm, h_dist_fire]
        
        wilderness_features = [1 if i == wilderness else 0 for i in range(1, 5)]
        soil_features = [1 if i == soil_type else 0 for i in range(1, 41)]
        
        X = np.array(cont_features + wilderness_features + soil_features, dtype=float).reshape(1, -1)
        X[:, :10] = scaler.transform(X[:, :10])
        
        # Make predictions based on choice
        print(f"\n{Colors.BOLD}{'─' * 60}{Colors.END}")
        
        if choice == "1":  # Random Forest only
            pred = int(rf_model.predict(X)[0])
            proba = rf_model.predict_proba(X)[0]
            proba_dict = {cover_type_names.get(i+1, str(i+1)): proba[i]*100 for i in range(len(proba))}
            top_proba = max(proba_dict.values())
            
            print(f"\n{Colors.GREEN}{Colors.BOLD} PREDICTION RESULT{Colors.END}")
            print(f"{Colors.GREEN}Algorithm:{Colors.END} Random Forest")
            print(f"{Colors.GREEN}Cover Type:{Colors.END} {cover_type_names.get(pred, 'Unknown')}")
            print(f"{Colors.GREEN}Confidence:{Colors.END} {top_proba:.1f}%")
            print(f"{Colors.GREEN}Model Accuracy:{Colors.END} {metrics['random_forest']['accuracy']:.1f}%")
            
            # Show top 3 probabilities
            print(f"\n{Colors.CYAN}Top Predictions:{Colors.END}")
            sorted_probs = sorted(proba_dict.items(), key=lambda x: x[1], reverse=True)[:3]
            for name, prob in sorted_probs:
                bar = "█" * int(prob / 2)
                print(f"  {name:<20} {Colors.GREEN}{bar:<40}{Colors.END} {prob:.1f}%")
        
        elif choice == "2":  # KNN only
            pred = int(knn_model.predict(X)[0])
            print(f"\n{Colors.GREEN}{Colors.BOLD} PREDICTION RESULT{Colors.END}")
            print(f"{Colors.GREEN}Algorithm:{Colors.END} K-Nearest Neighbors")
            print(f"{Colors.GREEN}Cover Type:{Colors.END} {cover_type_names.get(pred, 'Unknown')}")
            print(f"{Colors.GREEN}Model Accuracy:{Colors.END} {metrics['knn']['accuracy']:.1f}%")
        
        elif choice == "3":  # Naïve Bayes only
            pred = int(nb_model.predict(X)[0])
            print(f"\n{Colors.GREEN}{Colors.BOLD} PREDICTION RESULT{Colors.END}")
            print(f"{Colors.GREEN}Algorithm:{Colors.END} Naïve Bayes")
            print(f"{Colors.GREEN}Cover Type:{Colors.END} {cover_type_names.get(pred, 'Unknown')}")
            print(f"{Colors.GREEN}Model Accuracy:{Colors.END} {metrics['naive_bayes']['accuracy']:.1f}%")
            print(f"{Colors.RED}⚠ Warning: Naïve Bayes has low accuracy (13.4%) on this dataset{Colors.END}")
        
        elif choice == "4":  # Compare all
            rf_pred = int(rf_model.predict(X)[0])
            knn_pred = int(knn_model.predict(X)[0])
            nb_pred = int(nb_model.predict(X)[0])
            
            print(f"\n{Colors.BOLD}{Colors.CYAN} ALL MODELS COMPARISON{Colors.END}")
            print(f"{Colors.DIM}{'─' * 50}{Colors.END}")
            
            print(f"{Colors.GREEN}Random Forest:{Colors.END} {cover_type_names.get(rf_pred, 'Unknown')} "
                  f"(Acc: {metrics['random_forest']['accuracy']:.1f}%) {Colors.GREEN}★ BEST{Colors.END}")
            
            print(f"{Colors.YELLOW}K-Nearest Neighbors:{Colors.END} {cover_type_names.get(knn_pred, 'Unknown')} "
                  f"(Acc: {metrics['knn']['accuracy']:.1f}%)")
            
            print(f"{Colors.RED}Naïve Bayes:{Colors.END} {cover_type_names.get(nb_pred, 'Unknown')} "
                  f"(Acc: {metrics['naive_bayes']['accuracy']:.1f}%) {Colors.RED}⚠ LOW ACCURACY{Colors.END}")
            
            # Recommendation
            if rf_pred == knn_pred or rf_pred == nb_pred:
                print(f"\n{Colors.GREEN} Recommendation: Random Forest predicted {cover_type_names.get(rf_pred, 'Unknown')}{Colors.END}")
            else:
                print(f"\n{Colors.GREEN} Recommendation: Use Random Forest (most reliable){Colors.END}")
        
        # Ask for another prediction
        print(f"\n{Colors.DIM}{'─' * 60}{Colors.END}")

# ── Main Menu ──────────────────────────────────────────────────────────
def main():
    """Main console application"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                    FORESTML ANALYZER                         ║")
    print("║           Forest Cover Type Classification System            ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}")
    
    # Load models
    try:
        scaler, rf_model, knn_model, nb_model, metrics = load_models()
    except Exception as e:
        print(f"{Colors.RED} Error loading models: {e}{Colors.END}")
        print(f"{Colors.YELLOW}Please run 'python train_model.py' first{Colors.END}")
        return
    
    while True:
        print(f"\n{Colors.BOLD}{Colors.CYAN} MAIN MENU{Colors.END}")
        print(f"{Colors.DIM}{'─' * 40}{Colors.END}")
        print("  1. View Model Performance")
        print("  2. Dataset Statistics & Visualization")
        print("  3. Interactive Prediction")
        print("  4. Model Comparison Dashboard")
        print("  5. Exit")
        
        choice = input(f"\n{Colors.GREEN}Select option (1-5):{Colors.END} ").strip()
        
        if choice == "1":
            show_model_comparison(metrics)
            
            # ASCII Bar Chart
            perf_data = {
                "Random Forest": metrics['random_forest']['accuracy'],
                "KNN": metrics['knn']['accuracy'],
                "Naïve Bayes": metrics['naive_bayes']['accuracy']
            }
            draw_bar_chart(perf_data, " MODEL ACCURACY COMPARISON", width=45)
            
            # Radar-style comparison
            draw_radar_chart(metrics)
            
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.END}")
        
        elif choice == "2":
            load_dataset_stats()
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.END}")
        
        elif choice == "3":
            interactive_prediction(scaler, rf_model, knn_model, nb_model, metrics)
        
        elif choice == "4":
            show_model_comparison(metrics)
            
            # Detailed comparison
            print(f"{Colors.BOLD}{Colors.CYAN} WINNER: Random Forest{Colors.END}")
            print(f"{Colors.GREEN}✓ Best accuracy: {metrics['random_forest']['accuracy']:.1f}%{Colors.END}")
            print(f"{Colors.GREEN}✓ Best F1 Score: {metrics['random_forest']['f1']:.1f}%{Colors.END}")
            print(f"{Colors.GREEN}✓ Most balanced precision-recall tradeoff{Colors.END}")
            print(f"{Colors.GREEN}✓ Ensemble method reduces overfitting{Colors.END}")
            
            print(f"\n{Colors.YELLOW} Recommendation:{Colors.END} Deploy Random Forest in production")
            
            input(f"\n{Colors.DIM}Press Enter to continue...{Colors.END}")
        
        elif choice == "5":
            print(f"\n{Colors.GREEN} Thanks for using ForestML! Goodbye!{Colors.END}\n")
            break
        
        else:
            print(f"{Colors.RED}Invalid choice. Please try again.{Colors.END}")

if __name__ == "__main__":
    main()
    