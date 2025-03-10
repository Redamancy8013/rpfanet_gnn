def read_evaluation_results(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    car_metrics = {'bbox': [], 'bev': [], '3d': [], 'aos': []}
    for line_index, line in enumerate(lines):
        if 'Car AP@0.70, 0.50, 0.50:' in line:
            # 提取汽车类别的不同指标
            for metric in car_metrics.keys():
                for count in range(4):
                    metric_line = lines[line_index + count + 1]
                    if metric in metric_line:
                        for i in range(3):
                            metric_value = float(metric_line.split(':')[1].strip().split(',')[i])
                            car_metrics[metric].append(metric_value)

    return car_metrics

def calculate_average_metrics(metrics):
    if not metrics:
        return None
    average_metrics = {}
    for metric, values in metrics.items():
        average_metrics[metric] = [sum(values[i::3]) / len(values[i::3]) for i in range(3)]
    return average_metrics

root_path = "/home/ez/project/vod/output/"
file_name =[ "astyx_fsa.log", "astyx.log", "astyx_vel_fsa.log", "astyx_vel.log"]

for name in file_name:
    file_path = root_path + name
    car_metrics = read_evaluation_results(file_path)
    average_car_metrics = calculate_average_metrics(car_metrics)
    print(f"Average Car Metrics for {name}:", average_car_metrics)