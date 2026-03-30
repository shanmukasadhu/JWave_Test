python3 "/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Code/batch_cq500_hounsfield2density.py" \
  --dicom-root "/Users/shanmukasadhu/Downloads/500CTScans/qure.headct.study" \
  --out-root "/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density" \
  --max-scans 300 \
  --clean-head-mask 1 \
  --head-center-ellipse-frac 0.35

export CT_ACOUSTIC_DIR="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/acoustic_maps"
export CT_PROC_DIR="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/processed"
export CT_COMPARISON_OUT="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/ct_comparison_analysis"

python3 "/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Code/ct_scan_comparison_analysis.py"



export CT_ACOUSTIC_DIR="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/acoustic_maps"
export CT_PROC_DIR="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/processed"
export CT_COMPARISON_OUT="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/ct_comparison_analysis"
export CT_STRUCTURAL_OUT="/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Results/cq500_50_hounsfield2density/ct_structural_comparison"
python3 "/Users/shanmukasadhu/Documents/Jwave_Tests/ContinuousWave/Code/ct_scan_structural_comparison.py"


