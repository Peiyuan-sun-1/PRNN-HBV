# PRNN-HBV
====================================================================
Usage Guide and Important Notes for PRNN Models
====================================================================

1. Overview
-----------
This document provides the source code usage instructions for two Physics-informed Recurrent Neural Network (PRNN) models for hydrological simulation:

  - PRNN with Static Parameters: A baseline model where the 13 parameters of the underlying hydrological model (HBV) are treated as static, trainable variables. This model is implemented in the `PIMLHBV_Static.py` script.

  - PRNN-A (PRNN with Attention): An advanced model that incorporates a state-based attention mechanism into the hydrological simulation framework. This allows the model to dynamically learn the time-varying importance of different physical processes. This model is implemented in the `PIMLHBV_Attention.py` script.

These scripts are designed to read meteorological forcing data, train the respective models for specified basins, and evaluate their performance using the Nash-Sutcliffe Efficiency (NSE) metric.


2. System and Software Requirements
-----------------------------------
The models were developed and tested using a specific environment. To ensure full compatibility and reproducibility of our results, we strongly recommend using the following software versions:

* Python: 3.6
* TensorFlow: 2.13.0
* Keras: 2.13.1
* Pandas: 2.0.3
* NumPy: 1.24.3
* Matplotlib: 3.7.4
* h5py: 3.10.0

Installation:
It is highly recommended to use a virtual environment (e.g., venv or conda) to manage dependencies.

  # Create and activate a virtual environment (optional but recommended)
  python -m venv prnn_env
  source prnn_env/bin/activate  # On Windows, use `prnn_env\Scripts\activate`

  # Install the exact package versions for reproducibility
  pip install tensorflow==2.13.0 keras==2.13.1 pandas==2.0.3 numpy==1.24.3 matplotlib==3.7.4 h5py==3.10.0


3. Directory and Data Structure
-------------------------------
To run the scripts, your project directory should be organized as follows. The folder names are configurable within the scripts but are presented here with their default values.

/your_project_root/
|
+-- PIMLHBV_Static.py
+-- PIMLHBV_Attention.py
|
+-- basin_list.txt
+-- basin attributes.csv
|
+-- selected_basins_streamflow_csv/
|   +-- 01013500_streamflow_qc.csv
|   +-- 01022500_streamflow_qc.csv
|   +-- ...
|
+-- selected_basins_forcing_csv/
    +-- 01013500_lump_cida_forcing_leap.csv
    +-- 01022500_lump_cida_forcing_leap.csv
    +-- ...

Data Format Specifications:

* basin_list1.txt: A text file where each line corresponds to a basin. The script expects the basin ID to be the second element on each line (space-separated). The first line is treated as a header and skipped.

* Forcing Data CSVs: Each file should contain daily meteorological data for one basin. Essential columns are: 'Date', 'prcp(mm/day)', 'tmean(C)', and 'dayl(s)'.

* Streamflow Data CSVs: Each file should contain daily observed streamflow. Essential columns are 'date' and 'streamflow'. Missing values should be coded as -999.

* Attributes CSV ('basin attributes.csv'): A single file containing static attributes for all basins. It must contain 'gauge_id' and 'area_gages2' (watershed area in km²) columns to link basins and perform unit conversions.


4. How to Run the Models
------------------------
1. Prepare Data: Ensure your data files are structured and formatted as described in Section 3.

2. Configure Paths: Open the desired script (`PIMLHBV_Static.py` or `PIMLHBV_Attention.py`). In the `if __name__ == "__main__":` block at the bottom, verify that the folder and file paths match your directory structure.

3. Execute the Script: Run the script from your terminal. The program will iterate through the basin IDs listed in `basin_list1.txt`, processing one at a time.

   # To run the static PRNN model
   python PIMLHBV_Static.py

   # To run the PRNN-A model
   python PIMLHBV_Attention.py


5. Understanding the Output
---------------------------
Upon successful execution, a results folder (e.g., `PRNN_StaticOnly_Results` or `PRNN-A_Results_with_Analysis`) will be created containing the following files for each basin:

* {basin_id}_..._prnn.h5: A HDF5 file containing the learned weights of the best-performing model for that basin (saved based on validation set performance).

* ..._NSE_summary.csv: A summary CSV file compiling the best validation NSE and final test NSE for all processed basins.

* {basin_id}_analysis_output.csv (PRNN-A model only): A detailed daily time-series output for the test period, containing:
  - Date: The date of the record.
  - S1_... to S5_...: The five internal hydrological state variables.
  - att_w_S1 to att_w_S5: The learned attention weights corresponding to each state.
  - predicted_runoff_mm_final: The final, routed streamflow prediction from the model.
  - observed_runoff_mm: The actual observed streamflow.


6. Important Notes and Precautions
----------------------------------
* On Experimental Rigor: The provided scripts use a simple chronological split (e.g., 70% train, 30% test) for demonstration purposes. This is not the setup used for the final experiments in our paper. For robust scientific evaluation, we employed a 5-fold cross-validation scheme. Users intending to perform rigorous model validation should adapt the data splitting logic accordingly.

* Hyperparameter Tuning: The hyperparameters within the scripts (e.g., learning_rate=0.005, epochs=200) have been chosen based on our specific dataset. These values may not be optimal for other datasets or regions and should be considered starting points for further tuning.

* Warm-up Period (`WARMUP_PERIOD`): The models are stateful. The `WARMUP_PERIOD` (default: 365 days) is used to allow the model's internal states to reach a physically realistic condition before performance is evaluated. This is a critical step for obtaining meaningful results.

* Data Quality: The performance of any hydrological model is highly contingent on the quality of the input forcing and calibration data. Ensure your data are of high quality and have been properly pre-processed.

* Computational Resources: Training these models, especially for a large number of basins, can be computationally intensive. A GPU is strongly recommended to accelerate the training process.
