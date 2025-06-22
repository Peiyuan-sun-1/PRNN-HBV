# ==================================================================================================
#
# PIML-HBV Model with Static Parameters for Hydrological Simulation
#
# Abstract:
# This script implements a Physics-Informed Machine Learning (PIML) version of the
# Hydrologiska Byråns Vattenbalansavdelning (HBV) model. This specific version,

# termed PIMLHBV-Static, simplifies the original PIML-HBV framework by utilizing
# only static, trainable parameters. The dynamic parameter network and attention
# mechanism have been removed to create a more interpretable and computationally
# efficient baseline model. The model is designed to simulate streamflow for
# multiple basins, reading forcing data and basin attributes, training the model,
# and evaluating its performance using the Nash-Sutcliffe Efficiency (NSE) metric.
#
# ==================================================================================================


# ==================================================
# Section 1: Imports and Core Model Definition
# ==================================================
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Layer, Dense
from keras import initializers, callbacks
import keras.backend as K
from matplotlib import pyplot as plt
import traceback
import csv

# Define a global constant for the model's warmup period (in days).
# This period is excluded from loss and metric calculations to allow the model's
# internal states to stabilize before evaluation.
WARMUP_PERIOD = 365


def manual_triang(length):
    """
    Manually creates a triangular window for runoff routing.

    This function provides a custom implementation of a triangular window,
    ensuring compatibility across different TensorFlow versions where the
    built-in `tf.signal.triang` function might have varying behavior or availability.
    The window is used as a kernel for convolution to simulate the routing process.

    Args:
        length (tf.Tensor): The desired length of the triangular window (an integer cast to float).

    Returns:
        tf.Tensor: A 1D tensor of shape (length,) representing the triangular window.
    """
    length = tf.cast(length, tf.float32)
    # Handle the edge case of a single-point window.
    if tf.math.equal(length, 1.0):
        return tf.constant([1.0], dtype=tf.float32)
    # Generate a linear sequence from 0 to length-1.
    x = tf.range(length)
    # Calculate the midpoint of the window.
    midpoint = (length - 1) / 2
    # Create the triangular shape using an absolute difference from the midpoint.
    # K.epsilon() is added for numerical stability.
    window = 1.0 - tf.abs(x - midpoint) / (midpoint + K.epsilon())
    # Ensure all window values are non-negative.
    return tf.maximum(0.0, window)


class PIMLHBV_Static(Layer):
    """
    A custom Keras layer implementing the HBV hydrological model with static parameters.

    This layer encapsulates the core logic of the HBV model, a conceptual rainfall-runoff model.
    The term "Static" signifies that the model's 13 physical parameters are defined as
    single, trainable weights that remain constant throughout a simulation for a given basin.
    This contrasts with more complex PIML approaches where parameters might be dynamically
    predicted by a neural network.

    Attributes:
        mode (str): Determines the layer's output. 'normal' returns only the final simulated
                    streamflow. 'analysis' returns all internal states and fluxes, which is
                    useful for debugging and model interrogation.
        base_params (dict): A dictionary holding the trainable static parameters of the HBV model.
        base_maxbas (tf.Variable): A non-trainable parameter for the runoff routing kernel width.
    """

    def __init__(self, mode='normal', **kwargs):
        """
        Initializes the PIMLHBV_Static layer.

        Args:
            mode (str): The operational mode ('normal' or 'analysis'). Defaults to 'normal'.
            **kwargs: Standard Keras layer keyword arguments.
        """
        self.mode = mode
        super(PIMLHBV_Static, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Creates the layer's trainable weights (the HBV parameters).

        This method is called once by Keras to initialize the weights. It defines the 13
        core parameters of the HBV model as trainable variables. These are initialized to
        0.5 to start the optimization from a neutral point in the normalized parameter space.
        The `maxbas` parameter, which controls routing, is set as non-trainable here but
        could be made trainable if desired.

        Args:
            input_shape (tuple): The shape of the input tensor.
        """
        # Define the names of the 13 core HBV parameters.
        param_names = [
            'Tt', 'CFR', 'CFMAX', 'SCF', 'FC', 'Beta', 'LP', 'CWH',
            'UZL', 'K0', 'K1', 'K2', 'PER'
        ]
        self.base_params = {}
        # Create a trainable weight for each parameter.
        for name in param_names:
            self.base_params[name] = self.add_weight(
                name=f'base_{name}',
                shape=(1,),
                # Initialize in the middle of the normalized range [0, 1].
                initializer=initializers.Constant(value=0.5),
                trainable=True
            )
        # MAXBAS: a parameter for the triangular routing function. It is set as
        # non-trainable here, representing a fixed catchment response time.
        self.base_maxbas = self.add_weight(
            name='base_maxbas',
            shape=(1,),
            initializer=initializers.Constant(value=5.0),
            trainable=False
        )
        super(PIMLHBV_Static, self).build(input_shape)

    def heaviside(self, x):
        """
        A differentiable approximation of the Heaviside step function.

        The Heaviside function (H(x) = 1 if x > 0, else 0) is not differentiable at x=0,
        which poses a problem for gradient-based optimization. This function uses a
        scaled and shifted hyperbolic tangent (tanh) to create a smooth, differentiable
        step-like function.

        Args:
            x (tf.Tensor): The input tensor.

        Returns:
            tf.Tensor: The smoothed, step-like output.
        """
        return (K.tanh(5 * x) + 1) / 2

    def phyiscal_process(self, S1, S2, S3, S4, S5, P, T, PET, params):
        """
        Calculates one time step of the HBV model's physical processes.

        This function takes the current model states and meteorological inputs to compute
        the intermediate fluxes (e.g., snowmelt, evapotranspiration, runoff components)
        based on a given set of parameters.

        Args:
            S1-S5 (tf.Tensor): The five state variables of the HBV model (Snow Water Equivalent,
                               Snow Pack Water, Soil Moisture, Upper Zone Storage, Lower Zone Storage).
            P (tf.Tensor): Precipitation input for the current time step.
            T (tf.Tensor): Temperature input for the current time step.
            PET (tf.Tensor): Potential Evapotranspiration input for the current time step.
            params (list): A list of the 13 scaled HBV parameters.

        Returns:
            list: A list of the computed intermediate fluxes.
        """
        # Unpack the parameters.
        Tt, CFR, CFMAX, SCF, FC, Beta, LP, CWH, UZL, K0, K1, K2, PER = params

        # --- Parameter Scaling ---
        # The trainable parameters are normalized (approx. 0-1). Here, they are scaled
        # to physically plausible ranges for the HBV model equations.
        Tt = Tt * -1.5  # Threshold temperature (°C)
        CFR = CFR * 0.01  # Refreezing coefficient
        CFMAX = CFMAX * 1  # Degree-day factor (mm/°C/day)
        SCF = SCF * 1  # Snowfall correction factor
        FC = FC * 100  # Maximum soil moisture (mm)
        Beta = Beta * 8  # Shape parameter for effective rainfall
        LP = LP * 2.5  # Soil moisture threshold for ET reduction
        CWH = CWH * 0.3  # Water holding capacity of snowpack
        UZL = UZL * 120  # Threshold for rapid runoff from upper zone (mm)
        K0 = K0 * 0.5  # Recession coefficient for rapid runoff (1/day)
        K1 = K1 * 0.3  # Recession coefficient for interflow (1/day)
        K2 = K2 * 0.01  # Recession coefficient for baseflow (1/day)
        PER = PER * 3.5  # Percolation rate to lower zone (mm/day)

        # --- HBV Flux Calculations ---
        # Snow routine
        Ps = P * SCF * self.heaviside(Tt - T)  # Snowfall
        R_fr = K.minimum((Tt - T) * CFR * CFMAX, S2) * self.heaviside(Tt - T)  # Refreezing
        Pr = P * self.heaviside(T - Tt)  # Rainfall
        M = K.minimum((T - Tt) * CFMAX, S1) * self.heaviside(T - Tt)  # Snowmelt

        # Soil moisture routine
        I = self.heaviside(S2 - (S1 * CWH)) * (Pr + M)  # Infiltration
        R_e = I * ((S3 / FC) ** Beta)  # Effective rainfall (to runoff generation)
        E_a = self.heaviside(S3) * (PET * S3 / (FC * LP)) * self.heaviside((FC * LP) - S3) + \
              self.heaviside(S3) * PET * self.heaviside(S3 - (FC * LP))  # Actual evapotranspiration

        # Runoff generation routine
        Pe = PER * self.heaviside(S4)  # Percolation to lower zone
        q_0 = (S4 - UZL) * K0 * self.heaviside(S4 - UZL)  # Rapid runoff (surface)
        q_1 = S4 * K1 * self.heaviside(S4)  # Interflow
        q_2 = S5 * K2 * self.heaviside(S5)  # Baseflow

        return [Ps, R_fr, Pr, M, I, R_e, E_a, Pe, q_0, q_1, q_2]

    @tf.function
    def call(self, inputs):
        """
        Performs the forward pass of the layer, running the HBV simulation over a time series.

        This method iterates through the input time series day by day, updating the HBV model's
        internal states and calculating the daily runoff. The `@tf.function` decorator compiles
        this method into a static TensorFlow graph, significantly improving performance.

        Args:
            inputs (tf.Tensor): A tensor of shape (batch_size, seq_len, num_features)
                                containing Precipitation, Temperature, and PET.

        Returns:
            tf.Tensor: The simulated streamflow. If mode is 'analysis', returns a concatenation
                       of all states and fluxes.
        """
        # Extract meteorological forcing data from the input tensor.
        P_seq, T_seq, PET_seq = inputs[:, :, 0:1], inputs[:, :, 1:2], inputs[:, :, 2:3]
        batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

        # Initialize the five HBV state variables to zero for each sequence in the batch.
        S_t = tf.zeros((batch_size, 5), dtype=tf.float32)

        # Retrieve the static parameters. This list is passed to the physics function.
        params = [self.base_params[name] for name in
                  ['Tt', 'CFR', 'CFMAX', 'SCF', 'FC', 'Beta', 'LP', 'CWH', 'UZL', 'K0', 'K1', 'K2', 'PER']]

        # Use TensorArray for efficient and dynamically sized storage within the TF graph loop.
        q_total_ta = tf.TensorArray(tf.float32, size=seq_len)
        s_all_ta = tf.TensorArray(tf.float32, size=seq_len)

        # Main simulation loop over the time sequence.
        for t in tf.range(seq_len):
            # Unpack current states for clarity.
            S1, S2, S3, S4, S5 = [S_t[:, i:i + 1] for i in range(5)]

            # Calculate intermediate fluxes for the current time step `t`.
            Ps, R_fr, Pr, M, I, R_e, E_a, Pe, q_0, q_1, q_2 = self.phyiscal_process(
                S1, S2, S3, S4, S5, P_seq[:, t, :], T_seq[:, t, :], PET_seq[:, t, :], params
            )

            # --- State Update Equations (Euler forward integration) ---
            # Calculate the change (delta) for each state variable.
            _ds1 = Ps + R_fr - M  # Change in Snow Water Equivalent
            _ds2 = Pr + M - I - R_fr  # Change in Snow Pack Water
            _ds3 = I - E_a - R_e  # Change in Soil Moisture
            _ds4 = R_e - q_0 - q_1 - Pe  # Change in Upper Zone Storage
            _ds5 = Pe - q_2  # Change in Lower Zone Storage

            # Update states for the next time step.
            # `K.clip` prevents numerical instability from exploding gradients.
            # `tf.maximum` ensures states remain non-negative.
            next_S1 = tf.maximum(S1 + K.clip(_ds1, -1e5, 1e5), 0)
            # Snow pack water cannot exceed a fraction (CWH) of the snow water equivalent.
            next_S2 = tf.maximum(tf.minimum(S2 + K.clip(_ds2, -1e5, 1e5), next_S1 * params[7]), 0)
            next_S3 = tf.maximum(S3 + K.clip(_ds3, -1e5, 1e5), 0)
            next_S4 = tf.maximum(S4 + K.clip(_ds4, -1e5, 1e5), 0)
            next_S5 = tf.maximum(S5 + K.clip(_ds5, -1e5, 1e5), 0)

            # Concatenate updated states into a single tensor for the next iteration.
            S_t = tf.concat([next_S1, next_S2, next_S3, next_S4, next_S5], axis=1)

            # Sum the three runoff components to get the total runoff for the day.
            q_total = q_0 + q_1 + q_2

            # Write the daily results to their respective TensorArrays.
            q_total_ta = q_total_ta.write(t, q_total)
            s_all_ta = s_all_ta.write(t, S_t)

        # --- Post-loop Processing ---
        # Convert the TensorArrays to dense Tensors.
        # `stack` creates a tensor of shape (seq_len, batch_size, features).
        # `transpose` reorders it to the standard (batch_size, seq_len, features).
        q_final_seq = tf.transpose(q_total_ta.stack(), [1, 0, 2])
        s_all_seq = tf.transpose(s_all_ta.stack(), [1, 0, 2])

        # --- Runoff Routing ---
        # Apply a triangular routing function to smooth the generated runoff (q_final_seq).
        # This simulates the time delay and attenuation of runoff as it travels through
        # the river network to the basin outlet.
        maxbas_int = tf.cast(tf.round(self.base_maxbas), tf.int32)
        # The kernel width is derived from the MAXBAS parameter.
        kernel_width = tf.maximum(2 * maxbas_int[0] - 1, 1)

        # Create and normalize the triangular routing kernel.
        weights_unnorm = manual_triang(tf.cast(kernel_width, tf.float32))
        weights_norm = weights_unnorm / (tf.reduce_sum(weights_unnorm) + K.epsilon())
        # Reshape the kernel to be compatible with `tf.nn.conv1d`.
        weights_res = weights_norm[:, tf.newaxis, tf.newaxis]

        # Apply the routing via 1D convolution. 'SAME' padding ensures the output
        # sequence has the same length as the input.
        q_smoothed = tf.nn.conv1d(q_final_seq, weights_res, stride=1, padding='SAME')

        if self.mode == "normal":
            # In normal mode, return only the final, routed streamflow.
            return q_smoothed
        elif self.mode == "analysis":
            # In analysis mode, return all internal states and runoff components for inspection.
            return tf.concat([s_all_seq, q_final_seq, q_smoothed], axis=-1)


# ==================================================
# Section 2: Data and Utility Functions
# ==================================================
class DataforIndividual:
    """
    A class to handle loading and initial processing of data for a single basin.
    """

    def __init__(self, streamflow_folder, forcing_folder, basin_id, attributes_file):
        """
        Initializes the data loader with necessary file paths and basin identifier.

        Args:
            streamflow_folder (str): Path to the folder containing streamflow CSV files.
            forcing_folder (str): Path to the folder containing meteorological forcing CSV files.
            basin_id (str): The unique identifier for the basin.
            attributes_file (str): Path to the CSV file containing static basin attributes (e.g., area).
        """
        self.streamflow_folder = streamflow_folder
        self.forcing_folder = forcing_folder
        self.basin_id = basin_id.zfill(8)  # Ensure 8-digit format with leading zeros.
        self.attributes_file = attributes_file

    def load_data(self):
        """
        Loads, cleans, and merges forcing, streamflow, and attribute data for the basin.

        Returns:
            tuple: A tuple containing:
                - filtered_data (pd.DataFrame): Combined and filtered data.
                - selected_attributes (np.ndarray): Static attributes for the basin.
                - watershed_area_km2 (float): The basin's watershed area in square kilometers.
        """
        forcing_file = os.path.join(self.forcing_folder, f'{self.basin_id}_lump_cida_forcing_leap.csv')
        streamflow_file = os.path.join(self.streamflow_folder, f'{self.basin_id}_streamflow_qc.csv')

        # Load datasets
        forcing_data = pd.read_csv(forcing_file)
        streamflow_data = pd.read_csv(streamflow_file)
        attributes_data = pd.read_csv(self.attributes_file)

        # Find basin attributes, specifically the area needed for unit conversion.
        gauge_id = int(self.basin_id)
        basin_attributes = attributes_data[attributes_data['gauge_id'] == gauge_id]
        if basin_attributes.empty:
            raise ValueError(f"Gauge ID {self.basin_id} not found in attributes file.")
        watershed_area_km2 = basin_attributes['area_gages2'].values[0]

        # --- Data Cleaning and Merging ---
        forcing_data.columns = forcing_data.columns.str.strip()
        streamflow_data.columns = streamflow_data.columns.str.strip()
        streamflow_data.rename(columns={'streamflow': 'flow(cfs)', 'date': 'Date'}, inplace=True)

        # Convert 'Date' columns to datetime objects for proper merging and filtering.
        forcing_data['Date'] = pd.to_datetime(forcing_data['Date'])
        streamflow_data['Date'] = pd.to_datetime(streamflow_data['Date'])

        # Handle missing streamflow values (-999) and interpolate.
        streamflow_data['flow(cfs)'].replace(-999, pd.NA, inplace=True)
        streamflow_data['flow(cfs)'] = streamflow_data['flow(cfs)'].interpolate(method='linear').fillna(
            method='bfill').fillna(method='ffill')

        # Merge forcing and streamflow data on the 'Date' column.
        combined_data = pd.merge(forcing_data, streamflow_data, on='Date')

        # Filter the data to a specific, consistent time period for all basins.
        mask = (combined_data['Date'] >= '2000-01-01') & (combined_data['Date'] <= '2014-12-31')
        filtered_data = combined_data.loc[mask]

        selected_attributes = basin_attributes.drop(columns=['gauge_id', 'gauge_lat', 'gauge_lon']).values.flatten()

        return filtered_data, selected_attributes, watershed_area_km2


def preprocess_data(data, watershed_area_km2):
    """
    Converts streamflow units from cubic feet per second (cfs) to millimeters per day (mm/day).

    This conversion is essential to make the streamflow units consistent with precipitation units,
    allowing for direct comparison and proper calculation of water balance.

    Args:
        data (pd.DataFrame): The dataframe containing flow data in cfs.
        watershed_area_km2 (float): The basin's area in square kilometers.

    Returns:
        pd.DataFrame: The dataframe with streamflow converted to 'flow(mm)'.
    """
    data = data[['Date', 'dayl(s)', 'prcp(mm/day)', 'tmean(C)', 'flow(cfs)']].copy()
    # Convert cfs -> cubic meters per second (cms)
    data.loc[:, 'flow_cms'] = data['flow(cfs)'] * 0.0283168
    # Convert cms -> cubic meters per day
    data.loc[:, 'flow_m3_per_day'] = data['flow_cms'] * 86400
    # Convert volume (m^3/day) to depth (mm/day) by dividing by area (m^2) and converting m to mm.
    data.loc[:, 'flow(mm)'] = (data['flow_m3_per_day'] / (watershed_area_km2 * 1e6)) * 1e3
    data.drop(columns=['flow(cfs)', 'flow_cms', 'flow_m3_per_day'], inplace=True)
    return data


def calculate_pet(data):
    """
    Calculates Potential Evapotranspiration (PET) using a simplified temperature-based method.

    This function implements an empirical formula (similar to the Hargreaves method) to estimate
    PET from mean daily temperature and daylight hours.

    Args:
        data (pd.DataFrame): Dataframe containing 'tmean(C)' and 'dayl(s)'.

    Returns:
        pd.DataFrame: The dataframe with an added 'PET' column.
    """
    tmean = data['tmean(C)'].values
    dayl_fraction = data['dayl(s)'].values / 86400  # Convert daylight seconds to fraction of a day
    # Empirical formula for PET calculation
    pet = 29.8 * (dayl_fraction * 24) * 0.611 * np.exp(17.3 * tmean / (tmean + 237.3)) / (tmean + 273.2)
    data['PET'] = pet
    return data


def split_data(hydrodata):
    """
    Splits the hydrological data into training and testing sets.

    A 70/30 chronological split is used, where the first 70% of the data is used
    for training and the remaining 30% for testing.

    Args:
        hydrodata (pd.DataFrame): The complete preprocessed dataframe.

    Returns:
        tuple: A tuple of (train_set, test_set) as pandas DataFrames.
    """
    hydrodata = hydrodata[['Date', 'prcp(mm/day)', 'tmean(C)', 'PET', 'flow(mm)']]
    split_point = int(len(hydrodata) * 0.7)
    train_set = hydrodata.iloc[:split_point].copy()
    test_set = hydrodata.iloc[split_point:].copy()
    return train_set, test_set


def generate_train_test(train_set, test_set, wrap_length):
    """
    Generates training and testing samples in the format required by the model.

    The training data is segmented into multiple overlapping sequences of `wrap_length`.
    This technique of data augmentation is common for time-series models to create
    a larger, more diverse set of training examples. The test data is kept as a single,
    contiguous sequence.

    Args:
        train_set (pd.DataFrame): The training data.
        test_set (pd.DataFrame): The testing data.
        wrap_length (int): The length of each training sequence.

    Returns:
        tuple: A tuple of (train_x, train_y, test_x, test_y) as NumPy arrays.
    """
    # Extract model inputs (forcing) and outputs (streamflow)
    train_x_np = train_set[['prcp(mm/day)', 'tmean(C)', 'PET']].values
    train_y_np = train_set[['flow(mm)']].values
    test_x_np = test_set[['prcp(mm/day)', 'tmean(C)', 'PET']].values
    test_y_np = test_set[['flow(mm)']].values

    # Create overlapping sequences for training
    # A new sequence starts every 365 days.
    wrap_number_train = max(1, (train_set.shape[0] - wrap_length) // 365 + 1)
    train_x = np.zeros(shape=(wrap_number_train, wrap_length, train_x_np.shape[1]))
    train_y = np.zeros(shape=(wrap_number_train, wrap_length, 1))

    for i in range(wrap_number_train):
        start = i * 365
        end = start + wrap_length
        train_x[i, :, :] = train_x_np[start:end, :]
        train_y[i, :, :] = train_y_np[start:end, :]

    # The test set is treated as a single batch with one long sequence.
    test_x = np.expand_dims(test_x_np, axis=0)
    test_y = np.expand_dims(test_y_np, axis=0)

    return train_x, train_y, test_x, test_y


def calculate_nse(observed, simulated):
    """
    Calculates the Nash-Sutcliffe Efficiency (NSE).

    NSE is a standard metric in hydrology for evaluating the goodness-of-fit of a model.
    NSE = 1 corresponds to a perfect match. NSE = 0 means the model is only as good as the mean of the observed data.
    NSE < 0 means the observed mean is a better predictor than the model.

    Args:
        observed (np.ndarray): The array of observed streamflow values.
        simulated (np.ndarray): The array of simulated streamflow values.

    Returns:
        float: The NSE value.
    """
    observed_mean = np.mean(observed)
    numerator = np.sum((simulated - observed) ** 2)
    denominator = np.sum((observed - observed_mean) ** 2) + 1e-9  # Epsilon for stability
    return 1 - (numerator / denominator)


def nse_loss(y_true, y_pred):
    """
    Custom Keras loss function based on the NSE metric.

    The goal of training is to maximize NSE. Since optimizers minimize a loss function,
    we define the loss as `1 - NSE`. Minimizing this quantity is equivalent to maximizing NSE.
    The warmup period is excluded from the calculation.

    Args:
        y_true (tf.Tensor): The true (observed) values.
        y_pred (tf.Tensor): The predicted (simulated) values from the model.

    Returns:
        tf.Tensor: The calculated loss value.
    """
    # Exclude the warmup period from loss calculation.
    y_true_w = y_true[:, WARMUP_PERIOD:, :]
    y_pred_w = y_pred[:, WARMUP_PERIOD:, :]

    numerator = K.sum(K.square(y_pred_w - y_true_w), axis=1)
    denominator = K.sum(K.square(y_true_w - K.mean(y_true_w, axis=1, keepdims=True)), axis=1)

    # We want to minimize the ratio of error variance to observed variance.
    return K.mean(numerator / (denominator + K.epsilon()))


def nse_metrics(y_true, y_pred):
    """
    Custom Keras metric to monitor NSE during training.

    This function calculates the actual NSE (not 1 - NSE) for display and for use in
    callbacks like ModelCheckpoint and EarlyStopping. The warmup period is excluded.

    Args:
        y_true (tf.Tensor): The true (observed) values.
        y_pred (tf.Tensor): The predicted (simulated) values from the model.

    Returns:
        tf.Tensor: The calculated NSE metric.
    """
    # Exclude the warmup period from metric calculation.
    y_true_w = y_true[:, WARMUP_PERIOD:, :]
    y_pred_w = y_pred[:, WARMUP_PERIOD:, :]

    numerator = K.sum(K.square(y_pred_w - y_true_w), axis=1)
    denominator = K.sum(K.square(y_true_w - K.mean(y_true_w, axis=1, keepdims=True)), axis=1)

    # Return 1.0 minus the ratio to get the actual NSE value.
    return K.mean(1.0 - (numerator / (denominator + K.epsilon())))


def create_model(input_shape):
    """
    Creates and compiles the Keras model.

    This function wraps the PIMLHBV_Static layer in a Keras Model object.

    Args:
        input_shape (tuple): The shape of the input data (seq_len, num_features).

    Returns:
        keras.Model: The compiled Keras model.
    """
    x_input = Input(shape=input_shape, name='Input_Meteo')
    hydro_output = PIMLHBV_Static(mode='normal', name='Hydro_Static')(x_input)
    model = Model(inputs=x_input, outputs=hydro_output)
    return model


def train_model(model, train_x, train_y, ep_number, lrate, save_path):
    """
    Trains the PIML-HBV model.

    This function sets up the training process, including callbacks for saving the best
    model, early stopping, and learning rate reduction. It then compiles and fits the model.

    Args:
        model (keras.Model): The model to be trained.
        train_x (np.ndarray): Training input data.
        train_y (np.ndarray): Training target data.
        ep_number (int): The maximum number of epochs for training.
        lrate (float): The initial learning rate.
        save_path (str): The file path to save the best model weights.

    Returns:
        keras.callbacks.History: The training history object.
    """
    # Callback to save the model weights only when validation NSE improves.
    save = callbacks.ModelCheckpoint(save_path, verbose=1, save_best_only=True, monitor='val_nse_metrics', mode='max',
                                     save_weights_only=True)
    # Callback to stop training if validation NSE does not improve for 'patience' epochs.
    es = callbacks.EarlyStopping(monitor='val_nse_metrics', mode='max', verbose=1, patience=40, min_delta=0.001,
                                 restore_best_weights=True)
    # Callback to reduce the learning rate if validation NSE plateaus.
    reduce = callbacks.ReduceLROnPlateau(monitor='val_nse_metrics', factor=0.5, patience=20, verbose=1, mode='max',
                                         min_delta=0.001, min_lr=lrate / 100)
    # Callback to stop training if a NaN loss is encountered.
    tnan = callbacks.TerminateOnNaN()

    # Compile the model with the Adam optimizer and custom loss/metrics.
    model.compile(loss=nse_loss, metrics=[nse_metrics], optimizer=tf.keras.optimizers.Adam(learning_rate=lrate))

    # Fit the model to the training data.
    history = model.fit(train_x, train_y, epochs=ep_number, batch_size=32,
                        callbacks=[save, es, reduce, tnan],
                        validation_split=0.2, shuffle=True, verbose=2)

    return history


# ==================================================
# Section 3: Main Execution Block
# ==================================================
if __name__ == "__main__":
    # --- Configuration ---
    # Define paths to data and where results will be stored.
    basin_list_file = 'basin_list.txt'
    streamflow_folder = 'selected_basins_streamflow_csv'
    forcing_folder = 'selected_basins_forcing_csv'
    attributes_file = 'basin attributes.csv'
    results_folder = 'PRNN_StaticOnly_Results'
    os.makedirs(results_folder, exist_ok=True)

    # --- Load Basin List ---
    # Read the list of basin IDs to be processed from a text file.
    basin_ids = []
    try:
        with open(basin_list_file, 'r', encoding='utf-8-sig') as file:
            next(file)  # Skip header line
            for line in file:
                parts = line.split()
                if len(parts) > 1:
                    basin_ids.append(parts[1])
    except Exception as e:
        print(f"Error reading basin list: {e}")
        exit()

    # --- Main Processing Loop ---
    # Iterate through each basin, train a model, and evaluate its performance.
    nse_results = []
    for basin_id in basin_ids:
        basin_id_padded = basin_id.zfill(8)
        print(f"\nProcessing basin: {basin_id_padded}")
        try:
            # 1. Load and prepare data for the current basin.
            data_loader = DataforIndividual(streamflow_folder, forcing_folder, basin_id_padded, attributes_file)
            hydrodata, _, watershed_area_km2 = data_loader.load_data()
            hydrodata = preprocess_data(hydrodata, watershed_area_km2)
            hydrodata = calculate_pet(hydrodata)

            # 2. Split data and generate sequences for the model.
            train_set, test_set = split_data(hydrodata)
            wrap_length = 2190  # Use 6-year sequences for training (2190 days)
            train_x, train_y, test_x, test_y = generate_train_test(train_set, test_set, wrap_length)

            # Skip basin if there isn't enough data to create even one training sample.
            if train_x.shape[0] == 0:
                print(f"Not enough data to create training samples for basin {basin_id_padded}. Skipping.")
                continue

            # 3. Create and train the model.
            save_path = os.path.join(results_folder, f'{basin_id_padded}_static_prnn.h5')
            model = create_model((None, train_x.shape[2]))
            model.summary()
            history = train_model(model, train_x, train_y, ep_number=200, lrate=0.005, save_path=save_path)

            # 4. Evaluate the model on the test set.
            best_train_nse = max(history.history['val_nse_metrics'])
            print(f"Best Validation NSE: {best_train_nse:.4f}")

            # Load the best weights saved during training.
            model.load_weights(save_path)
            pred_y = model.predict(test_x)

            # Calculate NSE on the test period, excluding the warmup phase.
            observed_test_flow = test_y[0, WARMUP_PERIOD:, 0]
            predicted_test_flow = pred_y[0, WARMUP_PERIOD:, 0]
            test_nse = calculate_nse(observed_test_flow, predicted_test_flow)
            print(f"Test NSE: {test_nse:.4f}")

            # 5. Store the results.
            nse_results.append([basin_id_padded, best_train_nse, test_nse])

        except Exception as e:
            # Robust error handling to ensure the script continues even if one basin fails.
            print(f"Error processing basin {basin_id_padded}: {e}")
            traceback.print_exc()

    # --- Save Final Results ---
    # Write the summary of validation and test NSE scores for all basins to a CSV file.
    nse_output_file = os.path.join(results_folder, 'PRNN_StaticOnly_NSE_summary.csv')
    with open(nse_output_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Basin ID', 'Best Val NSE', 'Test NSE'])
        writer.writerows(nse_results)
    print(f'\nNSE results saved to {nse_output_file}')
