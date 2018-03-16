'''
We test how the speed difference of one dimensional Fourier transforms
scales compared to an ND FFT.

[seb@tn1057 ~]$ python2 test.py
(0.0001265406608581543, 4.190206527709961e-05, 3.019914651493599)
(0.042764151096343996, 0.0030129075050354002, 14.19364883418203)
(18.106536149978638, 1.812160348892212, 9.99168542731183)

[seb@tn1057 ~]$ python3 test.py
0.00014037950022611768 3.1666200084146115e-05 4.433102167392657
0.05472569015037152 0.003303307999885874 16.566935372742186
24.02259105179983 2.1675284233497223 11.082941655120283

[seb@tn1057 ~]$ intelpython2 test.py 
(0.005964446067810059, 5.2857398986816404e-05, 112.84032476319351)
(0.0029075026512145998, 0.0004975557327270508, 5.843571805069721)
(1.016000509262085, 0.45690919160842897, 2.223637711654523)

[seb@tn1057 ~]$ intelpython3 test.py
0.00011658575022011064 4.0917399746831504e-05 2.849295188390817
0.05802462874999037 0.0036666590996901503 15.824931408238557
20.50899286759959 1.8326582622998102 11.19084408124337
'''

import numpy as np
import timeit

N1 = 257
N2 = 256

tr = 20

x1 = np.random.randn(N1)
x2 = np.random.randn(N2)
x11 = np.random.randn(N1, N1)
x22 = np.random.randn(N2, N2)
x111 = np.random.randn(N1, N1, N1)
x222 = np.random.randn(N2, N2, N2)


def f1():
    np.fft.fft(x1)


def f2():
    np.fft.fft(x2)


def f11():
    np.fft.fftn(x11)


def f22():
    np.fft.fftn(x22)


def f111():
    np.fft.fftn(x111)


def f222():
    np.fft.fftn(x222)


t1 = timeit.timeit(f1, number=tr)/tr
t2 = timeit.timeit(f2, number=tr)/tr
t11 = timeit.timeit(f11, number=tr)/tr
t22 = timeit.timeit(f22, number=tr)/tr
t111 = timeit.timeit(f111, number=tr)/tr
t222 = timeit.timeit(f222, number=tr)/tr

print(t1, t2, t1 / t2)
print(t11, t22, t11 / t22)
print(t111, t222, t111 / t222)
