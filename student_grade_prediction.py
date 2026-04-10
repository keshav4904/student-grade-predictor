"""
Student Grade Prediction Model
================================
Generates synthetic student data, trains multiple ML models,
evaluates them, and outputs charts + a prediction report.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. GENERATE SYNTHETIC DATASET ──────────────────────────────────────────

def generate_dataset(n=500):
    study_hours    = np.random.normal(6, 2.5, n).clip(0, 14)
    attendance_pct = np.random.normal(78, 15, n).clip(20, 100)
    prev_gpa       = np.random.normal(2.8, 0.7, n).clip(0, 4)
    sleep_hours    = np.random.normal(7, 1.2, n).clip(3, 10)
    assignments    = np.random.normal(75, 18, n).clip(0, 100)
    extracurricular= np.random.randint(0, 4, n)               # 0-3 activities
    parent_edu     = np.random.choice(['None','HS','Bachelor','Graduate'],
                                       n, p=[0.15,0.35,0.35,0.15])
    internet       = np.random.choice([0, 1], n, p=[0.2, 0.8])
    stress_level   = np.random.choice(['Low','Medium','High'],
                                       n, p=[0.3, 0.4, 0.3])
    tutoring       = np.random.choice([0, 1], n, p=[0.6, 0.4])

    # Encode categoricals for target formula
    edu_map   = {'None': 0, 'HS': 1, 'Bachelor': 2, 'Graduate': 3}
    stress_map= {'Low': 0, 'Medium': -3, 'High': -7}
    edu_num   = np.array([edu_map[x] for x in parent_edu])
    stress_num= np.array([stress_map[x] for x in stress_level])

    # Target: weighted formula + noise
    grade = (
          study_hours   * 2.5
        + attendance_pct* 0.25
        + prev_gpa      * 8.0
        + sleep_hours   * 1.0
        + assignments   * 0.20
        + extracurricular * 1.5
        + edu_num       * 1.5
        + internet      * 2.0
        + stress_num
        + tutoring      * 3.0
        + np.random.normal(0, 4, n)
    )
    # Scale to 0–100
    grade = ((grade - grade.min()) / (grade.max() - grade.min()) * 100).clip(0, 100)

    df = pd.DataFrame({
        'study_hours':     study_hours,
        'attendance_pct':  attendance_pct,
        'prev_gpa':        prev_gpa,
        'sleep_hours':     sleep_hours,
        'assignments_avg': assignments,
        'extracurricular': extracurricular,
        'parent_education':parent_edu,
        'internet_access': internet,
        'stress_level':    stress_level,
        'tutoring':        tutoring,
        'final_grade':     grade
    })
    return df


# ── 2. PREPROCESSING ──────────────────────────────────────────────────────

def preprocess(df):
    df = df.copy()
    le = LabelEncoder()
    df['parent_education_enc'] = le.fit_transform(df['parent_education'])
    stress_map = {'Low': 0, 'Medium': 1, 'High': 2}
    df['stress_enc'] = df['stress_level'].map(stress_map)
    features = [
        'study_hours','attendance_pct','prev_gpa','sleep_hours',
        'assignments_avg','extracurricular','parent_education_enc',
        'internet_access','stress_enc','tutoring'
    ]
    X = df[features]
    y = df['final_grade']
    return X, y, features


# ── 3. TRAIN MODELS ───────────────────────────────────────────────────────

def train_and_evaluate(X, y, features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        'Linear Regression':     LinearRegression(),
        'Ridge Regression':      Ridge(alpha=1.0),
        'Decision Tree':         DecisionTreeRegressor(max_depth=6, random_state=42),
        'Random Forest':         RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42),
        'Gradient Boosting':     GradientBoostingRegressor(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42),
    }

    results = {}
    for name, model in models.items():
        use_scaled = name in ('Linear Regression', 'Ridge Regression')
        Xtr = X_train_s if use_scaled else X_train.values
        Xte = X_test_s  if use_scaled else X_test.values
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        cv    = cross_val_score(model, Xtr, y_train, cv=5, scoring='r2')
        results[name] = {
            'model':   model,
            'preds':   preds,
            'MAE':     mean_absolute_error(y_test, preds),
            'RMSE':    np.sqrt(mean_squared_error(y_test, preds)),
            'R2':      r2_score(y_test, preds),
            'CV_R2':   cv.mean(),
            'CV_std':  cv.std(),
        }

    return results, X_test, y_test, scaler, features


# ── 4. PLOTS ──────────────────────────────────────────────────────────────

PALETTE = {
    'teal':  '#1D9E75',
    'blue':  '#378ADD',
    'coral': '#D85A30',
    'amber': '#BA7517',
    'gray':  '#888780',
    'light': '#F1EFE8',
    'dark':  '#2C2C2A',
}

def make_plots(df, results, X_test, y_test, features, out_dir='/mnt/user-data/outputs'):
    X, y, _ = preprocess(df)
    best_name = max(results, key=lambda k: results[k]['R2'])
    best      = results[best_name]

    fig = plt.figure(figsize=(18, 20), facecolor='white')
    fig.suptitle('Student Grade Prediction — Model Report',
                 fontsize=22, fontweight='bold', color=PALETTE['dark'], y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    colors_list = [PALETTE['teal'], PALETTE['blue'], PALETTE['amber'],
                   PALETTE['coral'], PALETTE['gray']]
    model_names = list(results.keys())

    # ── A: Model Comparison (R²) ──
    ax1 = fig.add_subplot(gs[0, :2])
    r2s  = [results[n]['R2'] for n in model_names]
    bars = ax1.barh(model_names, r2s, color=colors_list, edgecolor='none', height=0.55)
    for bar, val in zip(bars, r2s):
        ax1.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', fontsize=11, color=PALETTE['dark'])
    ax1.set_xlim(0, 1.1)
    ax1.set_xlabel('R² Score', fontsize=11)
    ax1.set_title('Model Comparison — R² Score', fontsize=13, fontweight='bold', pad=10)
    ax1.axvline(0.8, color='#ccc', lw=1, linestyle='--')
    ax1.text(0.81, -0.5, 'good threshold', fontsize=9, color='#999')
    ax1.spines[['top','right','bottom']].set_visible(False)
    ax1.tick_params(left=False)

    # ── B: MAE / RMSE Table ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    tdata = [[n, f"{results[n]['MAE']:.2f}", f"{results[n]['RMSE']:.2f}", f"{results[n]['R2']:.3f}"]
             for n in model_names]
    tbl = ax2.table(cellText=tdata,
                    colLabels=['Model', 'MAE', 'RMSE', 'R²'],
                    loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#ddd')
        if r == 0:
            cell.set_facecolor(PALETTE['teal'])
            cell.set_text_props(color='white', fontweight='bold')
        elif model_names[r-1] == best_name if r > 0 and r-1 < len(model_names) else False:
            cell.set_facecolor('#E1F5EE')
        else:
            cell.set_facecolor('white')
    ax2.set_title('Metrics Summary', fontsize=12, fontweight='bold', pad=10)

    # ── C: Best model — Actual vs Predicted ──
    ax3 = fig.add_subplot(gs[1, 0])
    preds = best['preds']
    ax3.scatter(y_test, preds, alpha=0.5, color=PALETTE['teal'], s=25, edgecolors='none')
    mn, mx = min(y_test.min(), preds.min()), max(y_test.max(), preds.max())
    ax3.plot([mn,mx],[mn,mx], color=PALETTE['coral'], lw=1.5, linestyle='--')
    ax3.set_xlabel('Actual Grade', fontsize=10)
    ax3.set_ylabel('Predicted Grade', fontsize=10)
    ax3.set_title(f'{best_name}\nActual vs Predicted', fontsize=11, fontweight='bold')
    ax3.spines[['top','right']].set_visible(False)

    # ── D: Residuals distribution ──
    ax4 = fig.add_subplot(gs[1, 1])
    residuals = y_test.values - preds
    ax4.hist(residuals, bins=25, color=PALETTE['blue'], edgecolor='white', alpha=0.85)
    ax4.axvline(0, color=PALETTE['coral'], lw=1.5, linestyle='--')
    ax4.set_xlabel('Residual (Actual − Predicted)', fontsize=10)
    ax4.set_ylabel('Count', fontsize=10)
    ax4.set_title('Residuals Distribution', fontsize=11, fontweight='bold')
    ax4.spines[['top','right']].set_visible(False)

    # ── E: Feature Importance (Random Forest) ──
    ax5 = fig.add_subplot(gs[1, 2])
    rf = results['Random Forest']['model']
    importances = pd.Series(rf.feature_importances_, index=features).sort_values()
    bar_colors = [PALETTE['teal'] if v > importances.median() else PALETTE['gray']
                  for v in importances]
    importances.plot(kind='barh', ax=ax5, color=bar_colors, edgecolor='none')
    ax5.set_xlabel('Importance', fontsize=10)
    ax5.set_title('Feature Importance\n(Random Forest)', fontsize=11, fontweight='bold')
    ax5.spines[['top','right','bottom']].set_visible(False)
    ax5.tick_params(labelsize=9)

    # ── F: Grade Distribution ──
    ax6 = fig.add_subplot(gs[2, 0])
    ax6.hist(df['final_grade'], bins=30, color=PALETTE['amber'], edgecolor='white', alpha=0.85)
    ax6.set_xlabel('Final Grade', fontsize=10)
    ax6.set_ylabel('Students', fontsize=10)
    ax6.set_title('Grade Distribution', fontsize=11, fontweight='bold')
    ax6.spines[['top','right']].set_visible(False)

    # ── G: Study Hours vs Grade ──
    ax7 = fig.add_subplot(gs[2, 1])
    sc = ax7.scatter(df['study_hours'], df['final_grade'],
                     c=df['attendance_pct'], cmap='YlGn',
                     alpha=0.5, s=18, edgecolors='none')
    plt.colorbar(sc, ax=ax7, label='Attendance %', shrink=0.8)
    ax7.set_xlabel('Study Hours / Day', fontsize=10)
    ax7.set_ylabel('Final Grade', fontsize=10)
    ax7.set_title('Study Hours vs Grade\n(colour = attendance)', fontsize=11, fontweight='bold')
    ax7.spines[['top','right']].set_visible(False)

    # ── H: CV R² comparison ──
    ax8 = fig.add_subplot(gs[2, 2])
    cv_means = [results[n]['CV_R2'] for n in model_names]
    cv_stds  = [results[n]['CV_std'] for n in model_names]
    xpos = range(len(model_names))
    ax8.bar(xpos, cv_means, color=colors_list, edgecolor='none', width=0.6)
    ax8.errorbar(xpos, cv_means, yerr=cv_stds, fmt='none',
                 color=PALETTE['dark'], capsize=4, lw=1.5)
    ax8.set_xticks(xpos)
    short_names = [n.replace(' Regression','').replace(' Boosting','') for n in model_names]
    ax8.set_xticklabels(short_names, rotation=20, ha='right', fontsize=9)
    ax8.set_ylabel('CV R²', fontsize=10)
    ax8.set_title('Cross-validated R² ± std', fontsize=11, fontweight='bold')
    ax8.set_ylim(0, 1.05)
    ax8.spines[['top','right']].set_visible(False)

    plt.savefig(f'{out_dir}/student_grade_model_report.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[✓] Plot saved.")
    return best_name, best


# ── 5. PREDICT NEW STUDENT ────────────────────────────────────────────────

def predict_student(model, scaler, features, use_scaled=True, **kwargs):
    row = pd.DataFrame([kwargs])
    X   = row[features]
    Xs  = scaler.transform(X) if use_scaled else X.values
    return model.predict(Xs)[0]


# ── 6. MAIN ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("Generating dataset...")
    df = generate_dataset(500)

    print(f"Dataset: {df.shape[0]} students, {df.shape[1]} features")
    print(df.describe().round(2).to_string())

    X, y, features = preprocess(df)
    print("\nTraining models...")
    results, X_test, y_test, scaler, features = train_and_evaluate(X, y, features)

    print("\n── Model Performance ──")
    for name, r in results.items():
        print(f"  {name:<25}  R²={r['R2']:.3f}  MAE={r['MAE']:.2f}  RMSE={r['RMSE']:.2f}  CV-R²={r['CV_R2']:.3f}±{r['CV_std']:.3f}")

    print("\nGenerating plots...")
    best_name, best = make_plots(df, results, X_test, y_test, features)

    # Sample predictions
    print(f"\n── Best Model: {best_name}  (R²={best['R2']:.3f}) ──")
    print("\nSample predictions on 3 hypothetical students:")
    sample_students = [
        dict(study_hours=8, attendance_pct=92, prev_gpa=3.5, sleep_hours=7.5,
             assignments_avg=88, extracurricular=2, parent_education_enc=2,
             internet_access=1, stress_enc=0, tutoring=1),
        dict(study_hours=3, attendance_pct=55, prev_gpa=1.8, sleep_hours=5,
             assignments_avg=50, extracurricular=0, parent_education_enc=1,
             internet_access=0, stress_enc=2, tutoring=0),
        dict(study_hours=5, attendance_pct=75, prev_gpa=2.6, sleep_hours=7,
             assignments_avg=70, extracurricular=1, parent_education_enc=1,
             internet_access=1, stress_enc=1, tutoring=0),
    ]
    labels = ['High achiever', 'Struggling student', 'Average student']
    use_scaled = best_name in ('Linear Regression', 'Ridge Regression')
    for label, s in zip(labels, sample_students):
        pred = predict_student(best['model'], scaler, features, use_scaled=use_scaled, **s)
        grade_letter = 'A' if pred>=90 else 'B' if pred>=80 else 'C' if pred>=70 else 'D' if pred>=60 else 'F'
        print(f"  {label:<22}  Predicted grade: {pred:.1f}/100  ({grade_letter})")

    print("\n[✓] Done.")
