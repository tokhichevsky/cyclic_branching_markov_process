def runge_kutta(function, x0, y0, step, steps_num):
    y_arr = [y0]
    x_arr = [x0]

    x_end = x0 + step * steps_num

    while x_arr[-1] < x_end:
        x = x_arr[-1]
        y = y_arr[-1]

        k1 = function(x, y)
        k2 = function(x + step / 2, y + step / 2 * k1)
        k3 = function(x + step / 2, y + step / 2 * k2)
        k4 = function(x + step, y + step * k3)

        x_arr.append(x + step)
        y_arr.append(y + step / 6 * (k1 + 2 * k2 + 2 * k3 + k4))

    return x_arr, y_arr
