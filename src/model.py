import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Softmax, Multiply, GlobalAveragePooling1D
from tensorflow.keras.models import Model

def attention_layer(inputs):
    # Calculate attention weights
    a = Dense(1, activation='tanh')(inputs)
    a = Softmax(axis=1)(a)
    
    # Apply weights
    output = Multiply()([inputs, a])
    # Average pool over time dimension (serializable & standard)
    output = GlobalAveragePooling1D()(output)
    return output

def create_model(input_shape, num_classes=8):
    inputs = Input(shape=input_shape)
    
    # --- CNN Block ---
    x = Conv1D(64, kernel_size=3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Conv1D(128, kernel_size=3, padding='same')(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)
    
    # --- BiLSTM Block ---
    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = Dropout(0.4)(x)
    
    # --- Attention Layer ---
    x = attention_layer(x)
    
    # --- Fully Connected Layers ---
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # --- Output Layer ---
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    # Test model shape
    model = create_model((130, 182), 8)
    model.summary()
