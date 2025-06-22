# ==================================================================================================
#
# Physics-informed Recurrent Neural Network with Attention (PRNN-A) for Hydrological Simulation
#
# Abstract:
# This script implements an advanced Physics-Informed Machine Learning (PIML) model, termed
# PRNN-A, which enhances the core HBV hydrological model with a state-based self-attention
# mechanism. The model dynamically re-weights its internal state variables (e.g., snowpack,
# soil moisture) at each time step, allowing it to learn the relative importance of different
# physical stores under varying hydrological conditions. This approach aims to improve model
# performance and provide insights into the internal functioning of the hydrological system.
# The script is structured to train the model and then perform a detailed analysis by
# extracting and saving the internal states and attention weights over the test period.
#
# Disclaimer:
# This script is designed for demonstration purposes, illustrating the PRNN-A model's
# architecture and analysis capabilities on a single basin. The actual research experiments
# based on this model employ a more rigorous validation methodology, such as 5-fold
# cross-validation, which is not implemented in this demonstrative version.
#
# ==================================================================================================


# ==================================================
# Section 1: Imports and Core Model Definition
# ==================================================
import os
import traceback
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Layer, Dense
from keras import initializers, callbacks
import keras.backend as K

# Define a global constant for the model's warmup period (in days).
# This period is excluded from loss and metric calculations to allow the model's
# internal states to stabilize before evaluation.
WARMUP_PERIOD = 365


def manual_triang(length):
    """
    Manually creates a triangular window for runoff routing.

    This function provides a custom implementation of a triangular window,
    ensuring compatibility across different TensorFlow versions. The window is used
    as a convolution kernel to simulate the temporal smoothing and delay of runoff
    as it travels through the catchment.

    Args:
        length (tf.Tensor): The desired length of the triangular window.

    Returns:
        tf.Tensor: A 1D tensor representing the triangular window.
    """
    length = tf.cast(length, tf.float32)
    # Handle the edge case of a single-point window.
    if tf.math.equal(length, 1.0):
        return tf.constant([1.0], dtype=tf.float32)

    # Generate a linear sequence and calculate the midpoint.
    x = tf.range(length)
    midpoint = (length - 1) / 2

    # Create the triangular shape and ensure all values are non-negative.
    window = 1.0 - tf.abs(x - midpoint) / (midpoint + K.epsilon())
    return tf.maximum(0.0, window)


class PIMLHBV_Attention(Layer):
    """
    A custom Keras layer implementing the HBV model with a state-based attention mechanism (PRNN-A).

    This layer integrates a small neural network (attention mechanism) into the HBV
    simulation loop. At each time step, the attention network takes the current five
    hydrological states as input and outputs a set of weights. These weights are used to
    create an "effective" state vector, which is then used in the physical process
    calculations. This allows the model to dynamically prioritize certain physical
    processes (e.g., snowmelt vs. soil moisture) based on the system's current condition.

    Attributes:
        mode (str): Determines the layer's output. 'normal' for training (returns only
                    streamflow), 'analysis' for interpretation (returns states, attention
                    weights, and streamflow).
        alpha (float): A hyperparameter that controls the blending between the original
                       states and the attention-weighted states. A value of 1 would use
                       only attended states, while 0 would disable attention.
    """

    def __init__(self, mode='normal', alpha=0.5, **kwargs):
        """
        Initializes the PIMLHBV_Attention layer.

        Args:
            mode (str): The operational mode ('normal' or 'analysis'). Defaults to 'normal'.
            alpha (float): The blending factor for the attention mechanism. Defaults to 0.5.
            **kwargs: Standard Keras layer keyword arguments.
        """
        self.mode = mode
        self.alpha = alpha
        super(PIMLHBV_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Initializes all trainable weights for the layer, including both the physical
        parameters and the attention network weights.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        # Define the 13 core HBV parameters as trainable weights.
        param_names = ['Tt', 'CFR', 'CFMAX', 'SCF', 'FC', 'Beta', 'LP', 'CWH', 'UZL', 'K0', 'K1', 'K2', 'PER']
        self.base_params = {}
        for name in param_names:
            self.base_params[name] = self.add_weight(
                name=f'base_{name}', shape=(1,),
                initializer=initializers.RandomUniform(minval=0.1, maxval=0.9), trainable=True
            )
        # Define the non-trainable routing parameter.
        self.base_maxbas = self.add_weight(
            name='base_maxbas', shape=(1,),
            initializer=initializers.Constant(value=4.0), trainable=False
        )

        # Define the layers for the attention network (a small MLP).
        # This network learns to map the current state vector to attention scores.
        self.att_dense1 = Dense(32, activation='relu', name='attention_dense_1')
        self.att_dense2 = Dense(16, activation='relu', name='attention_dense_2')
        self.att_dense3 = Dense(5, activation=None, name='attention_scores')  # 5 scores for 5 states

        super(PIMLHBV_Attention, self).build(input_shape)

    def heaviside(self, x):
        """A smooth, differentiable approximation of the Heaviside step function."""
        return (K.tanh(5 * x) + 1) / 2

    def phyiscal_process(self, S1, S2, S3, S4, S5, P, T, PET, params):
        """
        Calculates one time step of the HBV model's physical processes using the
        (potentially attention-weighted) state variables.
        """
        # Unpack and scale the 13 normalized parameters to their physical ranges.
        Tt, CFR, CFMAX, SCF, FC, Beta, LP, CWH, UZL, K0, K1, K2, PER = params
        Tt = Tt * 4.0 - 1.5
        CFR = CFR * 0.1
        CFMAX = CFMAX * 9.0 + 1.0
        SCF = SCF * 0.6 + 0.4
        FC = FC * 50.0 + 50.0
        Beta = Beta * 5.0 + 1.0
        LP = LP * 0.7 + 0.3
        CWH = CWH * 0.2
        UZL = UZL * 70.0
        K0 = K0 * 0.4 + 0.1
        K1 = K1 * 0.39 + 0.01
        K2 = K2 * 0.149 + 0.001
        PER = PER * 3.0

        # --- HBV Flux Calculations ---
        # Snow routine
        Ps = P * SCF * self.heaviside(Tt - T)
        R_fr = K.minimum((Tt - T) * CFR * CFMAX, S2) * self.heaviside(Tt - T)
        Pr = P * self.heaviside(T - Tt)
        M = K.minimum((T - Tt) * CFMAX, S1) * self.heaviside(T - Tt)

        # Soil moisture routine
        I_potential = Pr + M
        # Excess water that cannot be held by the snowpack
        Ex = K.maximum(0., S2 + I_potential - S1 * CWH)
        I = I_potential - Ex  # Actual infiltration into soil
        R_e = (I + Ex) * ((S3 / FC) ** Beta)
        E_a = self.heaviside(S3) * (PET * S3 / (FC * LP)) * self.heaviside((FC * LP) - S3) + \
              self.heaviside(S3) * PET * self.heaviside(S3 - (FC * LP))

        # Runoff generation routine
        Pe = PER * self.heaviside(S4)
        q_0 = (S4 - UZL) * K0 * self.heaviside(S4 - UZL)
        q_1 = S4 * K1 * self.heaviside(S4)
        q_2 = S5 * K2 * self.heaviside(S5)
        return [Ps, R_fr, Pr, M, I, Ex, R_e, E_a, Pe, q_0, q_1, q_2]

    @tf.function
    def call(self, inputs):
        """
        Performs the forward pass, running the attention-augmented HBV simulation.
        The `@tf.function` decorator compiles this method into a static TensorFlow graph
        for significant performance improvement.
        """
        P_seq, T_seq, PET_seq = inputs[:, :, 0:1], inputs[:, :, 1:2], inputs[:, :, 2:3]
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

        # Initialize states and parameters
        S_t = tf.zeros((batch_size, 5), dtype=tf.float32)
        params = [self.base_params[name] for name in
                  ['Tt', 'CFR', 'CFMAX', 'SCF', 'FC', 'Beta', 'LP', 'CWH', 'UZL', 'K0', 'K1', 'K2', 'PER']]

        # Initialize TensorArrays to store time-series results efficiently in the graph.
        q_total_ta = tf.TensorArray(tf.float32, size=seq_len)
        s_all_ta = tf.TensorArray(tf.float32, size=seq_len)
        att_weights_ta = tf.TensorArray(tf.float32, size=seq_len)  # For storing attention weights

        # Main simulation loop
        for t in tf.range(seq_len):
            # --- Attention Mechanism ---
            # 1. Pass the current state vector through the attention MLP to get scores.
            att_h1 = self.att_dense1(S_t)
            att_h2 = self.att_dense2(att_h1)
            att_scores = self.att_dense3(att_h2)

            # 2. Mask scores for states that are zero to prevent attention on empty reservoirs.
            # This is a crucial step for physical consistency.
            mask = tf.cast(S_t > 1e-9, tf.float32)  # Create a binary mask for non-zero states
            masked_scores = att_scores * mask + (1.0 - mask) * -1e9  # Set scores of zero-states to -inf

            # 3. Apply softmax to get a probability distribution (weights) over the states.
            att_weights = tf.nn.softmax(masked_scores, axis=-1)
            att_weights_ta = att_weights_ta.write(t, att_weights)

            # 4. Create the effective state vector for the physics calculations.
            S_att = S_t * att_weights  # Apply attention weights
            S_combined = self.alpha * S_att + (1 - self.alpha) * S_t  # Blend with original states
            S1, S2, S3, S4, S5 = [S_t[:, i:i + 1] for i in range(5)]  # Original states for updates
            S1_eff, S2_eff, S3_eff, S4_eff, S5_eff = [S_combined[:, i:i + 1] for i in
                                                      range(5)]  # Effective states for physics

            # --- HBV Physics ---
            # Calculate fluxes using the effective (attention-blended) states.
            Ps, R_fr, Pr, M, I, Ex, R_e, E_a, Pe, q_0, q_1, q_2 = self.phyiscal_process(
                S1_eff, S2_eff, S3_eff, S4_eff, S5_eff,
                P_seq[:, t, :], T_seq[:, t, :], PET_seq[:, t, :], params
            )

            # --- State Updates (using original states and calculated deltas) ---
            _ds1 = Ps + R_fr - M
            _ds2 = Pr + M - I - Ex - R_fr
            _ds3 = I - E_a - R_e
            _ds4 = R_e - q_0 - q_1 - Pe
            _ds5 = Pe - q_2
            next_S1 = tf.maximum(S1 + K.clip(_ds1, -1e5, 1e5), 0)
            next_S2 = tf.maximum(tf.minimum(S2 + K.clip(_ds2, -1e5, 1e5), next_S1 * params[7]),
                                 0)  # Use scaled CWH from params
            next_S3 = tf.maximum(S3 + K.clip(_ds3, -1e5, 1e5), 0)
            next_S4 = tf.maximum(S4 + K.clip(_ds4, -1e5, 1e5), 0)
            next_S5 = tf.maximum(S5 + K.clip(_ds5, -1e5, 1e5), 0)
            S_t = tf.concat([next_S1, next_S2, next_S3, next_S4, next_S5], axis=1)

            # --- Store Outputs ---
            q_total = q_0 + q_1 + q_2
            q_total_ta = q_total_ta.write(t, q_total)
            s_all_ta = s_all_ta.write(t, S_t)

        # --- Final Processing ---
        # Convert TensorArrays to dense Tensors and transpose to (batch, time, features).
        q_final_seq = tf.transpose(q_total_ta.stack(), [1, 0, 2])
        s_all_seq = tf.transpose(s_all_ta.stack(), [1, 0, 2])
        att_weights_seq = tf.transpose(att_weights_ta.stack(), [1, 0, 2])

        # Apply triangular routing to the generated runoff.
        maxbas_int = tf.cast(tf.round(self.base_maxbas), tf.int32)
        kernel_width = tf.maximum(2 * maxbas_int[0] - 1, 1)
        weights_unnorm = manual_triang(tf.cast(kernel_width, tf.float32))
        weights_norm = weights_unnorm / (tf.reduce_sum(weights_unnorm) + K.epsilon())
        weights_res = weights_norm[:, tf.newaxis, tf.newaxis]
        q_smoothed = tf.nn.conv1d(q_final_seq, weights_res, stride=1, padding='SAME')

        # Return outputs based on the specified mode.
        if self.mode == "normal":
            return q_smoothed
        elif self.mode == "analysis":
            # For analysis, concatenate all internal variables for detailed inspection.
            # Output structure: 5 states, 5 att_weights, 1 raw_q, 1 final_q
            return tf.concat([s_all_seq, att_weights_seq, q_final_seq, q_smoothed], axis=-1)


# ==================================================
# Section 2: Data and Utility Functions
# Note: These are standard helper functions for data handling, unit conversion,
#       and performance metric calculation. They are identical to the ones used
#       in the static model script.
# ==================================================
class DataforIndividual:
    """A class to handle loading and initial processing of data for a single basin."""

    def __init__(self, streamflow_folder, forcing_folder, basin_id, attributes_file):
        self.streamflow_folder = streamflow_folder
        self.forcing_folder = forcing_folder
        self.basin_id = basin_id.zfill(8)
        self.attributes_file = attributes_file

    def load_data(self):
        forcing_file = os.path.join(self.forcing_folder, f'{self.basin_id}_lump_cida_forcing_leap.csv')
        streamflow_file = os.path.join(self.streamflow_folder, f'{self.basin_id}_streamflow_qc.csv')
        forcing_data = pd.read_csv(forcing_file)
        streamflow_data = pd.read_csv(streamflow_file)
        attributes_data = pd.read_csv(self.attributes_file)
        gauge_id = int(self.basin_id)
        basin_attributes = attributes_data[attributes_data['gauge_id'] == gauge_id]
        if basin_attributes.empty: raise ValueError(f"Gauge ID {self.basin_id} not found")
        watershed_area_km2 = basin_attributes['area_gages2'].values[0]
        forcing_data.columns = forcing_data.columns.str.strip()
        streamflow_data.columns = streamflow_data.columns.str.strip()
        streamflow_data.rename(columns={'streamflow': 'flow(cfs)', 'date': 'Date'}, inplace=True)
        forcing_data['Date'] = pd.to_datetime(forcing_data['Date'])
        streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'])
        streamflow_data['flow(cfs)'].replace(-999, pd.NA, inplace=True)
        streamflow_data['flow(cfs)'] = streamflow_data['flow(cfs)'].interpolate(method='linear').fillna(
            method='bfill').fillna(method='ffill')
        combined_data = pd.merge(forcing_data, streamflow_data, on='Date')
        mask = (combined_data['Date'] >= '2000-01-01') & (combined_data['Date'] <= '2014-12-31')
        filtered_data = combined_data.loc[mask]
        selected_attributes = basin_attributes.drop(columns=['gauge_id', 'gauge_lat', 'gauge_lon']).values.flatten()
        return filtered_data, selected_attributes, watershed_area_km2


def preprocess_data(data, watershed_area_km2):
    """Converts streamflow from cfs to mm/day using watershed area."""
    data = data[['Date', 'dayl(s)', 'prcp(mm/day)', 'tmean(C)', 'flow(cfs)']].copy()
    data.loc[:, 'flow_cms'] = data['flow(cfs)'] * 0.0283168
    data.loc[:, 'flow_m3_per_day'] = data['flow_cms'] * 86400
    data.loc[:, 'flow(mm)'] = (data['flow_m3_per_day'] / (watershed_area_km2 * 1e6)) * 1e3
    data.drop(columns=['flow(cfs)', 'flow_cms', 'flow_m3_per_day'], inplace=True)
    return data


def calculate_pet(data):
    """Calculates Potential Evapotranspiration (PET) from temperature and daylight hours."""
    tmean = data['tmean(C)'].values
    dayl = data['dayl(s)'].values / 86400
    pet = 29.8 * (dayl * 24) * 0.611 * np.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)
    data['PET'] = pet
    return data


def split_data(hydrodata):
    """Splits data into 70% training and 30% testing sets chronologically."""
    hydrodata = hydrodata[['Date', 'prcp(mm/day)', 'tmean(C)', 'PET', 'flow(mm)']]
    split_point = int(len(hydrodata) * 0.7)
    train_set = hydrodata.iloc[:split_point].copy()
    test_set = hydrodata.iloc[split_point:].copy()
    return train_set, test_set


def generate_train_test(train_set, test_set, wrap_length):
    """Generates overlapping training sequences and a single test sequence."""
    train_x_np = train_set[['prcp(mm/day)', 'tmean(C)', 'PET']].values
    train_y_np = train_set[['flow(mm)']].values
    test_x_np = test_set[['prcp(mm/day)', 'tmean(C)', 'PET']].values
    test_y_np = test_set[['flow(mm)']].values
    wrap_number_train = max(1, (train_set.shape[0] - wrap_length) // 365 + 1)
    train_x = np.zeros(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.zeros(shape=(wrap_number_train, wrap_length, 1))
    for i in range(wrap_number_train):
        start = i * 365
        end = start + wrap_length
        train_x[i, :, :] = train_x_np[start:end, :]
        train_y[i, :, :] = train_y_np[start:end, :]
    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)
    return train_x, train_y, test_x, test_y


def calculate_nse(observed, simulated):
    """Calculates the Nash-Sutcliffe Efficiency (NSE)."""
    observed_mean = np.mean(observed)
    return 1 - (np.sum((simulated - observed) ** 2) / (np.sum((observed - observed_mean) ** 2) + 1e-9))


def nse_loss(y_true, y_pred):
    """Custom Keras loss function based on 1 - NSE."""
    y_true_w = y_true[:, WARMUP_PERIOD:, :]
    y_pred_w = y_pred[:, WARMUP_PERIOD:, :]
    numerator = K.sum(K.square(y_pred_w - y_true_w), axis=1)
    denominator = K.sum(K.square(y_true_w - K.mean(y_true_w, axis=1, keepdims=True)), axis=1)
    return K.mean(numerator / (denominator + K.epsilon()))


def nse_metrics(y_true, y_pred):
    """Custom Keras metric to monitor NSE directly during training."""
    y_true_w = y_true[:, WARMUP_PERIOD:, :]
    y_pred_w = y_pred[:, WARMUP_PERIOD:, :]
    numerator = K.sum(K.square(y_pred_w - y_true_w), axis=1)
    denominator = K.sum(K.square(y_true_w - K.mean(y_true_w, axis=1, keepdims=True)), axis=1)
    return K.mean(1.0 - (numerator / (denominator + K.epsilon())))


# ==================================================
# Section 3: Model Creation and Training
# ==================================================
def create_prnn_a_model(input_shape, mode='normal'):
    """
    Factory function to create the PRNN-A model.

    This function allows the creation of the model in two modes:
    - 'normal': Standard mode for training, outputs only the final streamflow.
    - 'analysis': For post-hoc analysis, outputs a detailed set of internal variables.

    Args:
        input_shape (tuple): The shape of the input data (seq_len, num_features).
        mode (str): The operational mode of the PIMLHBV_Attention layer.

    Returns:
        keras.Model: The constructed Keras model.
    """
    x_input = Input(shape=input_shape, name='Input_Meteo')
    hydro_output = PIMLHBV_Attention(mode=mode, name='Hydro_Attention')(x_input)
    model = Model(inputs=x_input, outputs=hydro_output)
    return model


def train_model(model, train_x, train_y, ep_number, lrate, save_path):
    """Compiles and trains the model with a set of standard callbacks."""
    save = callbacks.ModelCheckpoint(save_path, verbose=1, save_best_only=True, monitor='val_nse_metrics', mode='max',
                                     save_weights_only=True)
    es = callbacks.EarlyStopping(monitor='val_nse_metrics', mode='max', verbose=1, patience=40, min_delta=0.001,
                                 restore_best_weights=True)
    reduce = callbacks.ReduceLROnPlateau(monitor='val_nse_metrics', factor=0.5, patience=20, verbose=1, mode='max',
                                         min_delta=0.001, min_lr=lrate / 100)
    tnan = callbacks.TerminateOnNaN()
    model.compile(loss=nse_loss, metrics=[nse_metrics], optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))
    history = model.fit(train_x, train_y, epochs=ep_number, batch_size=32, callbacks=[save, es, reduce, tnan],
                        validation_split=0.2, shuffle=True, verbose=2)
    return history


# ==================================================
# Section 4: Main Execution Block for Demonstration
# ==================================================
if __name__ == "__main__":
    # --- Configuration ---
    basin_list_file = 'basin_list1.txt'
    streamflow_folder = '569selected_basins_streamflow_csv'
    forcing_folder = '569流域selected_basins_forcing_csv'
    attributes_file = '569流域属性.csv'
    results_folder = 'PRNN-A_Results_with_Analysis'
    os.makedirs(results_folder, exist_ok=True)

    # --- Load list of basins to process ---
    basin_ids = []
    try:
        with open(basin_list_file, 'r', encoding='utf-8-sig') as file:
            next(file)  # Skip header
            for line in file:
                parts = line.split()
                if len(parts) > 1: basin_ids.append(parts[1])
    except Exception as e:
        print(f"Error reading basin list file '{basin_list_file}': {e}")
        exit()

    # --- Main processing loop for each basin ---
    nse_results = []
    for basin_id in basin_ids:  # This script demonstrates by running one basin after another
        basin_id_padded = basin_id.zfill(8)
        print(f"\n{'=' * 20} Processing Basin: {basin_id_padded} {'=' * 20}")
        try:
            # 1. Load and preprocess data for the current basin.
            data_loader = DataforIndividual(streamflow_folder, forcing_folder, basin_id_padded, attributes_file)
            hydrodata, _, watershed_area_km2 = data_loader.load_data()
            hydrodata = preprocess_data(hydrodata, watershed_area_km2)
            hydrodata = calculate_pet(hydrodata)
            train_set, test_set = split_data(hydrodata)
            wrap_length = 2190  # Use 6-year sequences
            train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, wrap_length)
            if train_x.shape[0] == 0:
                print(f"WARNING: Not enough data for basin {basin_id_padded}. Skipping.")
                continue

            # 2. Create and train the model in 'normal' mode for efficiency.
            save_path = os.path.join(results_folder, f'{basin_id_padded}_prnn_a_model.h5')
            training_model = create_prnn_a_model((None, train_x.shape[2]), mode='normal')
            training_model.summary()
            history = train_model(training_model, train_x, train_y, ep_number=200, lrate=0.005, save_path=save_path)
            best_val_nse = max(history.history.get('val_nse_metrics', [0]))
            print(f"Best Validation NSE for basin {basin_id_padded}: {best_val_nse:.4f}")

            # 3. Evaluate the trained model on the test set to get the final NSE score.
            training_model.load_weights(save_path)  # Load best weights
            pred_y = training_model.predict(test_x)
            observed_test_flow = test_y[0, WARMUP_PERIOD:, 0]
            predicted_test_flow = pred_y[0, WARMUP_PERIOD:, 0]
            test_nse = calculate_nse(observed_test_flow, predicted_test_flow)
            print(f"Final Test NSE for basin {basin_id_padded}: {test_nse:.4f}")
            nse_results.append([basin_id_padded, best_val_nse, test_nse])

            # 4. --- DETAILED ANALYSIS ---
            # Create a new model in 'analysis' mode and load the same trained weights.
            # This allows us to extract internal states and attention weights without
            # the overhead during training.
            print("Creating and running analysis model to save states and weights...")
            analysis_model = create_prnn_a_model((None, test_x.shape[2]), mode='analysis')
            analysis_model.load_weights(save_path)
            analysis_output = analysis_model.predict(test_x)

            # Slice the analysis output tensor according to its defined structure.
            # Structure: 5 states, 5 att_weights, 1 raw_q, 1 final_q (12 columns total)
            s_states = analysis_output[0, :, 0:5]
            att_weights = analysis_output[0, :, 5:10]
            q_raw = analysis_output[0, :, 10]
            q_final = analysis_output[0, :, 11]

            # Assemble the detailed results into a pandas DataFrame for easy saving.
            dates = test_set['Date'].values
            analysis_df = pd.DataFrame({
                'Date': dates,
                'S1_snowpack': s_states[:, 0], 'S2_snowmelt_water': s_states[:, 1],
                'S3_soil_moisture': s_states[:, 2], 'S4_upper_zone': s_states[:, 3],
                'S5_lower_zone': s_states[:, 4], 'att_w_S1': att_weights[:, 0],
                'att_w_S2': att_weights[:, 1], 'att_w_S3': att_weights[:, 2],
                'att_w_S4': att_weights[:, 3], 'att_w_S5': att_weights[:, 4],
                'predicted_runoff_mm_raw': q_raw, 'predicted_runoff_mm_final': q_final,
                'observed_runoff_mm': test_y[0, :, 0]
            })

            # Save the analysis data to a CSV file.
            analysis_filename = os.path.join(results_folder, f'{basin_id_padded}_analysis_output.csv')
            analysis_df.to_csv(analysis_filename, index=False, float_format='%.4f')
            print(f"Analysis results with states and weights saved to: {analysis_filename}")

        except Exception as e:
            print(f"ERROR processing basin {basin_id_padded}: {e}")
            traceback.print_exc()

    # --- Save Final Summary Results ---
    # Write the summary of NSE scores for all processed basins to a CSV file.
    nse_output_file = os.path.join(results_folder, 'PRNN-A_NSE_summary.csv')
    try:
        with open(nse_output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Basin ID', 'Best_Validation_NSE', 'Test_NSE'])
            writer.writerows(nse_results)
        print(f"\nNSE summary results saved to {nse_output_file}")
    except Exception as e:
        print(f"ERROR saving summary file: {e}")