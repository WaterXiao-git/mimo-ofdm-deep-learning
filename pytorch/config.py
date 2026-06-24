"""
配置文件 - 集中管理所有系统参数
"""

# 系统参数
SYSTEM_CONFIG = {
    # 上行链路参数
    'NUM_UPLINK_SUBCARRIERS': 96,
    'NUM_UPLINK_SYMBOLS': 1,
    
    # 下行链路参数
    'NUM_DOWNLINK_DATA_SUBCARRIERS': 144,
    'NUM_DOWNLINK_DATA_SYMBOLS': 1,
    'NUM_DOWNLINK_CTRL_BITS': 5,
    'NUM_DOWNLINK_TX': 32,
    'NUM_DOWNLINK_RX': 4,
    
    # 其他参数
    'SEPARATE_REAL_IMAG': 2,
    'NUM_MAX_BITS': 144 * 1 * 32,  # NUM_DOWNLINK_DATA_SUBCARRIERS * NUM_DOWNLINK_DATA_SYMBOLS * 32
    
    # SNR范围
    'SNR_DL_RANGE': [-20, 20],
    'SNR_UL_RANGE': [-20, 0],
}

# 模型架构参数
MODEL_CONFIG = {
    'HIDDEN_DIM': 256,
    'ATTENTION_REDUCTION': 16,
    'DROPOUT_RATE': 0.1,
    'NUM_SCHEMES': 3,
}

# 训练参数
TRAINING_CONFIG = {
    'BATCH_SIZE': 32,
    'NUM_TRAINING_ITERATIONS': 1000,
    'LEARNING_RATE': 1e-3,
    'WEIGHT_DECAY': 1e-5,
    'PATIENCE': 50,
    'GRADIENT_CLIP_NORM': 1.0,
    
    # 学习率调度
    'USE_COSINE_ANNEALING': True,
    'USE_REDUCE_LR_ON_PLATEAU': False,
    
    # 损失函数参数
    'FOCAL_LOSS_ALPHA': 1.0,
    'FOCAL_LOSS_GAMMA': 2.0,
}

# 评估参数
EVALUATION_CONFIG = {
    'NUM_TESTING_ITERATIONS': 1000,
    'EVAL_BATCH_SIZE': 16,
    'SNR_EVAL_POINTS': list(range(-15, 16, 5)),
    'NUM_TESTS_PER_SNR': 100,
}

# 文件路径
PATHS = {
    'DATA_TRAIN': './data_train/H_train.npy',
    'DATA_TRAIN_ALT': '../data_train/H_train.npy',
    'MODEL_SAVE_DIR': './modelSubmit',
    'LOGS_DIR': './logs',
    'EVAL_RESULTS_DIR': './evaluation_results',
}

# 设备配置
DEVICE_CONFIG = {
    'USE_CUDA_IF_AVAILABLE': True,
    'RANDOM_SEED': 42,
}