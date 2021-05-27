import random
import math
import numpy as np
import matplotlib.pyplot as plt
from numerical_methods import runge_kutta

###########

T1 = 1.5
T2 = 0.042
T3 = 0.167
T4 = 0.25
T5 = 0.031
q23 = 0.65
q24 = 1 - q23

p_ans = [0.848, 0.024, 0.061, 0.049, 0.018]

p_start = np.matrix([1, 0, 0, 0, 0]).transpose()

kolmogorov_matrix = np.matrix(
    [
        [-1 / T1, 0, 0, 0, 1 / T5],
        [1 / T1, -1 / T2, 0, 0, 0],
        [0, q23 / T2, -1 / T3, 0, 0],
        [0, q24 / T2, 0, -1 / T4, 0],
        [0, 0, 1 / T3, 1 / T4, -1 / T5]
    ]
)

intensity_matrix = [
    [0, 1 / T1, 0, 0, 0],
    [0, 0, q23 / T2, q24 / T2, 0],
    [0, 0, 0, 0, 1 / T3],
    [0, 0, 0, 0, 1 / T4],
    [1 / T5, 0, 0, 0, 0]
]


def model(t, p):
    return kolmogorov_matrix.dot(p)


def show_plot(t_array, p_array):
    fig = plt.figure(figsize=(10, 10), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    for i in range(len(p_array)):
        ax.plot(t_array, p_array[i])

    for p in p_ans:
        plt.scatter(t_array[-1], p, s=10)

    plt.xticks(np.arange(0, t_array[-1]))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.show()


def show_double_plot(t1_array, p1_array, t2_array, p2_array):
    fig = plt.figure(figsize=(15, 15), dpi=80)
    ax = fig.add_subplot(1, 1, 1)

    colors = ["red", "green", "orange", "blue", "black"]

    for i in range(len(p1_array)):
        label = "p" + str(i + 1)
        ax.plot(t1_array, p1_array[i], color=colors[i], label=label)

    for i in range(len(p2_array)):
        label = "sim: p" + str(i + 1)
        ax.plot(t2_array, p2_array[i], '--', color=colors[i], label=label)

    for i, p in enumerate(p_ans):
        label = "exact: p" + str(i + 1)
        plt.scatter(t1_array[-1], p, color=colors[i], label=label, s=12)

    plt.xticks(np.arange(0, t1_array[-1]))
    plt.yticks(np.arange(0, 1, 0.05))
    plt.legend(loc='best')
    plt.show()


def calculate_time_interval(intensity):
    # print(math.log(random.random()))
    return -1 / intensity * math.log(random.random())


def simulation_modeling(t_max):
    p_arr = [[0 for p in p_ans]]
    # p_arr[0][0] = 1
    t_arr = [0]
    current_step = 0
    current_state = 0
    while sum(t_arr) < t_max:
        min_time_interval = math.inf
        min_time_interval_index = None
        for index, intensity in enumerate(intensity_matrix[current_state]):
            if intensity == 0:
                continue

            current_time_interval = calculate_time_interval(intensity)
            if min_time_interval > current_time_interval:
                min_time_interval_index = index
                min_time_interval = current_time_interval

        # print(current_state, min_time_interval_index, min_time_interval)
        current_p_nums = p_arr[-1].copy()
        current_p_nums[current_state] += min_time_interval
        p_arr.append(current_p_nums)
        current_state = min_time_interval_index
        current_step += 1
        t_arr.append(min_time_interval)

    average_p = []

    total_t = [0]
    for p, t in zip(p_arr[1:], t_arr[1:]):
        total_t.append(total_t[-1] + t)
        average_p.append([p_el / total_t[-1] for p_el in p])

    return total_t[1:], average_p


def two_arrays_sort(positions, arrays):
    def key(element):
        return element[0]

    sortable_array = [(positions[i], arrays[i]) for i in range(len(positions))]
    sortable_array.sort(key=key)

    result_positions = []
    result_arrays = []

    for element in sortable_array:
        result_positions.append(element[0])
        result_arrays.append(element[1])

    return result_positions, result_arrays


def multi_simulation_modeling(t_max, goes, interval_nums):
    t_arrs = []
    p_arrs = []
    for i in range(goes):
        t_arr, p_arr = simulation_modeling(t_max)
        t_arrs += t_arr
        p_arrs += p_arr
    step = int(len(t_arrs) / interval_nums)

    t_sorted_arrs, p_sorted_arrs = two_arrays_sort(t_arrs, p_arrs)

    result_p = [p_start]
    result_t = [0]

    for i in range(step, len(t_sorted_arrs), step):
        result_p.append(list(np.median(p_sorted_arrs[i - step:i], axis=0)))
        result_t.append(np.median(t_sorted_arrs[i - step:i], axis=0))

    return result_t, result_p


def handle_rk_y_array(y_arr):
    p_array = [[] for i in range(np.shape(y_arr[0])[0])]
    for y in y_arr:
        for i, element in enumerate(y):
            p_array[i].append(float(element))

    return p_array


if __name__ == '__main__':
    t_arr, y_arr = runge_kutta(model, 0, p_start, 0.05, 1000)
    p_arr = handle_rk_y_array(y_arr)
    show_plot(t_arr, p_arr)
    print(p_ans, "\n", y_arr[-1].transpose())

    # sim_t_arr, sim_p_arr = simulation_modeling(200)
    # show_plot(sim_t_arr, handle_rk_y_array(sim_p_arr))
    sim_t_arr, sim_y_arr = multi_simulation_modeling(50, 1000, 50)
    sim_p_arr = handle_rk_y_array(sim_y_arr)
    show_plot(sim_t_arr, sim_p_arr)
    print(sim_y_arr[-1])

    show_double_plot(t_arr, p_arr, sim_t_arr, sim_p_arr)
