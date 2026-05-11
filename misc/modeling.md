---

### MODEL TRAINING SPECIFICATION: SPATIO-TEMPORAL CLIMATE PREDICTION

**TARGET OBJECTIVE:** Train a predictive Deep Learning model to forecast Egypt's regional temperatures using time-series EarthAccess NetCDF data.

#### 1. PRIMARY ARCHITECTURE: ConvLSTM (Convolutional LSTM)

* **Rationale:** Standard LSTMs flatten spatial relationships. CNNs lack temporal awareness. ConvLSTM processes sequences of 2D grids, maintaining both the geographic heat distribution across Egypt and the chronological seasonality.
* **Expected Input Tensor Shape:** `(batch_size, time_steps, channels, latitude, longitude)`
* *Note:* `channels` will likely be 1 (temperature), unless we inject elevation or humidity later.


* **Expected Output:** A predicted 2D spatial grid `(channels, latitude, longitude)` representing the forecasted temperature map for the next time step(s).
* **Layer Structure:** 2-3 ConvLSTM2D layers followed by Batch Normalization, feeding into a 3D Convolutional layer to collapse the sequence into the final spatial prediction.

#### 2. FALLBACK ARCHITECTURE: Bi-Directional LSTM + Attention

*(Execute ONLY if the ConvLSTM 3D tensor pipeline proves too computationally heavy or complex to build by the deadline).*

* **Data Requirement:** The spatial grids must be geographically averaged into a 1D array prior to ingestion.
* **Expected Input Tensor Shape:** `(batch_size, time_steps, features)`
* **Mechanism:** Wrap the LSTM layers in a Bi-Directional wrapper to process seasonal trends forward and backward. Implement a custom Attention layer to weight extreme historical anomalies (e.g., specific heatwaves) over standard baseline days.

#### 3. COMPILATION & HYPERPARAMETERS

* **Loss Function:** **Huber Loss (Smooth L1 Loss)**.
* *Justification:* Climate data contains extreme, unpredictable outliers. Mean Squared Error (MSE) will skew the weights too aggressively toward these anomalies. Huber Loss remains robust against outlier heat spikes while converging smoothly on standard days.


* **Optimizer:** **AdamW** (Adam with Weight Decay). Set initial learning rate to `1e-3` or `5e-4` with a learning rate scheduler (e.g., `ReduceLROnPlateau`).
* **Sequence Length (Window Size):** Start with a 7-day or 14-day lookback window (`time_steps`) to predict the next `t+1` interval.

#### 4. REQUIRED DELIVERABLES (FOR SUNDAY'S METRICS)

1. **Saved Model Weights:** (`.pth` or `.h5` file).
2. **Evaluation Metrics:** Output standard MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error) on a holdout test set.
3. **Loss Curves:** A plot showing Training Loss vs. Validation Loss to prove the model isn't heavily overfitting to the 15-day chunks.
4. **Comparison**: Comparing to real data of a predicted timeline.
