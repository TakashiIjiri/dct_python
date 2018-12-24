# -*- coding: utf-8 -*-

import numpy as np
import math
import cv2


#dctIIの素朴な実装
# Fk = Σ(n=0:N-1)( fn cos( pi k (n+0.5) / N))
# input : f:np.array,
# output: F:np.array,
def my_DCTII( f ) :
    num = f.shape[0]
    F   = np.zeros_like(f)
    c   = math.pi / num
    for k in range(num) :
        for n in range(num) :
            F[k] += f[n] * math.cos(c*k*(n+0.5))
    return F

#dctIIIの素朴な実装
# Fk = f0/2 + Σ(n=1:N-1)( fn cos( pi n (k+0.5) / N))
# input : f:np.array,
# output: F:np.array,
def my_DCTIII( f ) :
    num = f.shape[0]
    F   = np.zeros_like(f)
    c   = math.pi / num
    for k in range(num) :
        F[k] += f[0]/2
        for n in range(1, num) :
            F[k] += f[n] * math.cos(c*n*(k+0.5))
    return F


#cv2.dctの素朴な実装
# Fk = sqrt(1/N) * Σ(n=0:N-1)( fn cos( pi k (n+0.5) / N))     for k==0
# Fk = sqrt(2/N) * Σ(n=0:N-1)( fn cos( pi k (n+0.5) / N))     for k==1
# input : f:np.array,
# output: F:np.array,
def my_cv2DCT( f ) :
    num = f.shape[0]
    F   = np.zeros_like(f)
    c   = math.pi / num
    for k in range(num) :
        for n in range(0, num) :
            F[k] += f[n] * math.cos(c*k*(n+0.5))
    F[0]     *= math.sqrt(1/num)
    F[1:num] *= math.sqrt(2/num)
    return F


def my_cv2DCT_inv( F ) :
    num = F.shape[0]
    f   = np.zeros_like(F)
    c   = math.pi / num
    coef = math.sqrt(2/num)
    for k in range(num) :
        f[k] += math.sqrt(1/num) * F[0]
        for n in range(1, num) :
            f[k] += coef * F[n] * math.cos(c*n*(k+0.5))
    return f




if __name__ == '__main__':

    #テスト用の信号を作成
    N = 10
    f = np.array(np.random.rand(N))

    #例1) DCTIIの逆離散コサイン変換は, DCTIIIの2/N倍
    f_dct2      = my_DCTII (f)
    f_dct2_dct3 = (2/N) * my_DCTIII(f_dct2)
    print("\n\n例１ DCTII(f) の逆変換は DCTIIIの2/N倍になる")
    print("f\n", f)
    print("DCTII(f)\n", f_dct2)
    print("(2/N)DCTIII(DCTII(f))\n", f_dct2_dct3)
    print(f - f_dct2_dct3) # zeroになる

    #例2) DCTIIIの逆離散コサイン変換は, DCTIIの2/N倍
    f_dct3      = my_DCTIII (f)
    f_dct3_dct2 = (2/N) * my_DCTII(f_dct3)
    print("\n\n例2 DCTIII(f) の逆変換は DCTIIの2/N倍になる")
    print("f\n", f)
    print("DCTIII(f)\n", f_dct3)
    print("(2/N)DCTII(DCTIII(f))\n", f_dct3_dct2)
    print(f - f_dct3_dct2) # zeroになる

    #例3) cv2.dctも自作してみる（DCTIIとDCTIIIに近いが係数と定数項が少し違う）
    #see https://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#dct
    print("\n\n例3 cv2のDCTを自作する")
    cv2_DCT  = cv2.dct(f).T
    f_myCV2DCT = my_cv2DCT(f)
    print("cv2.dct\n" , cv2_DCT )
    print("mycv2dct\n",f_myCV2DCT)
    print("f\n", f )
    print("dct_inv(dct(f))\n", my_cv2DCT_inv( f_myCV2DCT) )

    print(0)
