def sum_squares_cy(double[:] arr):
    cdef Py_ssize_t i, n = arr.shape[0]
    cdef double s = 0.0
    for i in range(n):
        s += arr[i] * arr[i]   # чистый C-цикл, без питоновских объектов
    return s