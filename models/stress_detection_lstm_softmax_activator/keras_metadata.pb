
�_root"_tf_keras_sequential*�^{"name": "sequential_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "sequential_20", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_60_input"}}, {"class_name": "LSTM", "config": {"name": "lstm_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_40", "trainable": true, "dtype": "float32", "alpha": 0.5}}, {"class_name": "LSTM", "config": {"name": "lstm_61", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_41", "trainable": true, "dtype": "float32", "alpha": 0.5}}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LSTM", "config": {"name": "lstm_62", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_42", "trainable": true, "dtype": "float32", "alpha": 0.5}}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}}, {"class_name": "LSTM", "config": {"name": "lstm_63", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "softmax"}}]}, "shared_object_id": 30, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 31}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 5]}, "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 60, 5]}, "float32", "lstm_60_input"]}], {}]}, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 60, 5]}, "float32", "lstm_60_input"]}, "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential_20", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "lstm_60_input"}, "shared_object_id": 0}, {"class_name": "LSTM", "config": {"name": "lstm_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 5}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_40", "trainable": true, "dtype": "float32", "alpha": 0.5}, "shared_object_id": 6}, {"class_name": "LSTM", "config": {"name": "lstm_61", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 11}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_41", "trainable": true, "dtype": "float32", "alpha": 0.5}, "shared_object_id": 12}, {"class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 13}, {"class_name": "LSTM", "config": {"name": "lstm_62", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 18}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_42", "trainable": true, "dtype": "float32", "alpha": 0.5}, "shared_object_id": 19}, {"class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 20}, {"class_name": "LSTM", "config": {"name": "lstm_63", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 25}, {"class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28}, {"class_name": "Activation", "config": {"name": "activation_18", "trainable": true, "dtype": "float32", "activation": "softmax"}, "shared_object_id": 29}]}}, "training_config": {"loss": "sparse_categorical_crossentropy", "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 32}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}2
�root.layer_with_weights-0"_tf_keras_rnn_layer*�{"name": "lstm_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 5]}, "stateful": false, "must_restore_from_config": false, "class_name": "LSTM", "config": {"name": "lstm_60", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 60, 5]}, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 5, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 5]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 31}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 5]}}2
�root.layer-1"_tf_keras_layer*�{"name": "leaky_re_lu_40", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_40", "trainable": true, "dtype": "float32", "alpha": 0.5}, "shared_object_id": 6}2
�root.layer_with_weights-1"_tf_keras_rnn_layer*�
{"name": "lstm_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTM", "config": {"name": "lstm_61", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 11, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 33}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 128]}}2
�root.layer-3"_tf_keras_layer*�{"name": "leaky_re_lu_41", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_41", "trainable": true, "dtype": "float32", "alpha": 0.5}, "shared_object_id": 12}2
�root.layer-4"_tf_keras_layer*�{"name": "dropout_20", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_20", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 13}2
�root.layer_with_weights-2"_tf_keras_rnn_layer*�
{"name": "lstm_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTM", "config": {"name": "lstm_62", "trainable": true, "dtype": "float32", "return_sequences": true, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 18, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 34}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 128]}}2
�root.layer-6"_tf_keras_layer*�{"name": "leaky_re_lu_42", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_42", "trainable": true, "dtype": "float32", "alpha": 0.5}, "shared_object_id": 19}2
�root.layer-7"_tf_keras_layer*�{"name": "dropout_21", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout_21", "trainable": true, "dtype": "float32", "rate": 0.3, "noise_shape": null, "seed": null}, "shared_object_id": 20}2
�	root.layer_with_weights-3"_tf_keras_rnn_layer*�
{"name": "lstm_63", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTM", "config": {"name": "lstm_63", "trainable": true, "dtype": "float32", "return_sequences": false, "return_state": false, "go_backwards": false, "stateful": false, "unroll": false, "time_major": false, "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 25, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, null, 128]}, "ndim": 3, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 35}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 60, 128]}}2
�
root.layer_with_weights-4"_tf_keras_layer*�{"name": "dense_20", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_20", "trainable": true, "dtype": "float32", "units": 6, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 26}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 27}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 28, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}2
�
�root.layer_with_weights-0.cell"_tf_keras_layer*�{"name": "lstm_cell_60", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell_60", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 4}2
�root.layer_with_weights-1.cell"_tf_keras_layer*�{"name": "lstm_cell_61", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell_61", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 10}2
�*root.layer_with_weights-2.cell"_tf_keras_layer*�{"name": "lstm_cell_62", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell_62", "trainable": true, "dtype": "float32", "units": 128, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 17}2
�8root.layer_with_weights-3.cell"_tf_keras_layer*�{"name": "lstm_cell_63", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "LSTMCell", "config": {"name": "lstm_cell_63", "trainable": true, "dtype": "float32", "units": 64, "activation": "tanh", "recurrent_activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 21}, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"gain": 1.0, "seed": null}, "shared_object_id": 22}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 23}, "unit_forget_bias": true, "kernel_regularizer": null, "recurrent_regularizer": null, "bias_regularizer": null, "kernel_constraint": null, "recurrent_constraint": null, "bias_constraint": null, "dropout": 0.0, "recurrent_dropout": 0.0, "implementation": 2}, "shared_object_id": 24}2
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 37}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 32}2