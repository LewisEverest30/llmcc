
def predict_diviation(test_result_file_path: str):
    """
    Returns the diviation between the predicted and the actual values.
    """
    test_result = []
    with open(test_result_file_path, "r") as f:
        for line in f:
            if not line:
                continue
            predict = float(line.split()[1])
            actual = float(line.split()[0])
            test_result.append((actual, predict))

    total_divitaion_with_error = 0  # 考虑预测出现较大错误的情况
    total_divitaion_without_error = 0  # 不考虑预测出现较大错误的情况
    times_with_error = 0
    times_without_error = 0
    div_less_than_10_percent_with_error = 0
    div_between_10_and_30_percent_with_error = 0
    div_more_than_30_percent_with_error = 0    
    div_less_than_10_percent_without_error = 0
    div_between_10_and_30_percent_without_error = 0
    div_more_than_30_percent_without_error = 0
    for result in test_result:
        try:
            div = abs((result[1] - result[0])) / abs(result[0])  # diviation = |predicted - actual| / actual
            if div < 0:
                print(f"{result[1]} {result[0]} {div}")
            if div <= 0.1:
                total_divitaion_without_error += div
                times_without_error += 1
                if div < 0.1:
                    div_less_than_10_percent_without_error += 1
                elif div < 0.3:
                    div_between_10_and_30_percent_without_error += 1
                else:
                    div_more_than_30_percent_without_error += 1

            total_divitaion_with_error += div 
            times_with_error += 1
            if div < 0.1:
                div_less_than_10_percent_with_error += 1
            elif div < 0.3:
                div_between_10_and_30_percent_with_error += 1
            else:
                div_more_than_30_percent_with_error += 1
        except ZeroDivisionError:
            print(f"{result[1]} {result[0]}")
    return {
        "error_rate": (times_with_error - times_without_error)/times_with_error,
        "mean_diviation_with_error": total_divitaion_with_error/times_with_error,
        "mean_diviation_without_error": total_divitaion_without_error/times_without_error,
        "diviation_distribution_with_error": [div_less_than_10_percent_with_error/times_with_error, div_between_10_and_30_percent_with_error/times_with_error, div_more_than_30_percent_with_error/times_with_error],
        "diviation_distribution_without_error": [div_less_than_10_percent_without_error/times_without_error, div_between_10_and_30_percent_without_error/times_without_error, div_more_than_30_percent_without_error/times_without_error]
    }


if __name__ == '__main__':

    result_file = [
        "output/test/predict_result_2025-04-01-15-31-59.txt",
        # "output/train/predict_result_2025-03-26-16-35-57.txt",
        "output/baselines/netllm/train/predict_result_2025-03-10-19-57-28.txt",
        "output/baselines/nlp/predict_result_ybc.txt",
    ]

    for file in result_file:
        result = predict_diviation(file)
        print(file)
        print(f"Error rate: {result['error_rate']}")
        print(f"Mean diviation: {result['mean_diviation_with_error']}, {result['mean_diviation_without_error']}")
        print(f"Diviation distribution with error: {result['diviation_distribution_with_error']}")
        print(f"Diviation distribution withour error: {result['diviation_distribution_without_error']}")
        print("\n\n")
