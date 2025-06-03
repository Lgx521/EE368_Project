from ultralytics import YOLO

# 初始化模型（使用预训练权重）
model = YOLO('yolov8s.pt')

# 修正后的配置
config = {
    'data': 'configs/chess_config.yaml',
    'epochs': 300,       # 增加训练轮次
    'imgsz': 640,        # 降低分辨率
    'batch': 16,         # 增大批次大小
    'device': 'cpu',  # 强制使用GPU
    'project': 'runs/chess_piece',
    'name': 'chess_piece_exp1',
    'optimizer': 'SGD',  # 改用SGD优化器 
    'lr0': 0.01,
    'lrf': 0.1,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    # 'patience': 50,      # 早停耐心值
    'save_period': 5,    # 模型保存频率
    'half': True         # 启用FP16加速
}

# 开始训练
results = model.train(**config)