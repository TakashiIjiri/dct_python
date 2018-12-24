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



#DCT 2D 横方向にDCTを計算してから縦方向にDCTを計算する
def my_DCT2D( fxy ) :
    #横方向にDCTIIをかけてから，縦方向にDCTIIをかける
    tmp = np.zeros(fxy.shape, dtype=np.float64)
    for i in range(fxy.shape[0]) : tmp[i] = my_DCTII(fxy[i])

    tmp = tmp.T
    Fuv = np.zeros(tmp.shape, dtype=float)
    for i in range(tmp.shape[0]) : Fuv[i] = my_DCTII(tmp[i])
    return Fuv.T

def my_DCT2D_inv( Fuv ) :
    #横方向にDCTIIIをかけてから，縦方向にDCTIIIをかける
    tmp = np.zeros(Fuv.shape, dtype=float)
    for i in range(Fuv.shape[0]) : tmp[i] = my_DCTIII(Fuv[i])
    tmp *= 2 / tmp.shape[1]

    tmp = tmp.T
    fxy = np.zeros(tmp.shape, dtype=float)
    for i in range(tmp.shape[0]) : fxy[i] = my_DCTIII(tmp[i])
    fxy *= 2 / fxy.shape[1]
    return fxy.T


#cv2.dct の2次元版を自作してみる
def my_cv2DCT2D( fxy ) :
    #横方向にDCTIIをかけてから，縦方向にDCTIIをかける
    tmp = np.zeros(fxy.shape, dtype=float)
    for i in range(fxy.shape[0]) : tmp[i] = my_cv2DCT(fxy[i])

    tmp = tmp.T
    Fuv = np.zeros(tmp.shape, dtype=float)
    for i in range(tmp.shape[0]) : Fuv[i] = my_cv2DCT(tmp[i])
    return Fuv.T

def my_cv2DCT2D_inv( Fuv ) :
    #横方向にDCTIIIをかけてから，縦方向にDCTIIIをかける
    tmp = np.zeros(Fuv.shape, dtype=float)
    for i in range(Fuv.shape[0]) : tmp[i] = my_cv2DCT_inv(Fuv[i])

    tmp = tmp.T
    fxy = np.zeros(tmp.shape, dtype=float)
    for i in range(tmp.shape[0]) : fxy[i] = my_cv2DCT_inv(tmp[i])
    return fxy.T





if __name__ == '__main__':

    #テスト用の画像を取得
    img = cv2.imread("img.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float64(img)

    #自作のdct2Dをテスト
    img_dct     = my_DCT2D(img)
    img_dct_inv = my_DCT2D_inv(img_dct)
    cv2.imwrite("ex1_img_dct.png"   , np.uint8( np.fabs(img_dct)* 0.1 ) )
    cv2.imwrite("ex1_img_dctinv.png", np.uint8( img_dct_inv) )

    #自作のcv2dct2Dをテスト
    img_dct     = my_cv2DCT2D(img)
    img_dct_inv = my_cv2DCT2D_inv(img_dct)
    cv2.imwrite("ex2_img_dct.png"   , np.uint8( np.fabs(img_dct)* 5.0 ) )
    cv2.imwrite("ex2_img_dctinv.png", np.uint8( img_dct_inv) )
    print( np.sum(np.abs(cv2.dct(img) - img_dct)) )

    #cv2.dctのテスト
    img_dct = cv2.dct(img)
    img_dct_inv = cv2.idct(img_dct)
    cv2.imwrite("ex3_img_dct.png"   , np.uint8( np.fabs(img_dct)* 5.0 ) )
    cv2.imwrite("ex3_img_dctinv.png", np.uint8( img_dct_inv) )
