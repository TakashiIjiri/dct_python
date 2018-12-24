# -*- coding: utf-8 -*-
import numpy as np
import math
import cv2


if __name__ == '__main__':

    #テスト用の画像を取得
    img = cv2.imread("img1.png")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.float64(img)

    H, W = img.shape[0], img.shape[1]
    #calc dct
    img_dct = cv2.dct(img)

    img_dct_1 = np.zeros_like(img_dct)
    img_dct_2    = np.zeros_like(img_dct)
    img_dct_4    = np.zeros_like(img_dct)
    img_dct_8    = np.zeros_like(img_dct)
    img_dct_1    = img_dct
    img_dct_2[0:H//2, 0:W//2] = img_dct[0:H//2, 0:W//2]
    img_dct_4[0:H//4, 0:W//4] = img_dct[0:H//4, 0:W//4]
    img_dct_8[0:H//8, 0:W//8] = img_dct[0:H//8, 0:W//8]

    cv2.imwrite("img_noComp.png", np.uint8( img   ) )
    #左上の1/1, 1/2, 1/4, 1/8のみを使った離散コサイン画像を出力
    cv2.imwrite("img_comp_1_.png", np.uint8( np.fabs(img_dct_1*10.0 )) )
    cv2.imwrite("img_comp_2_.png", np.uint8( np.fabs(img_dct_2*10.0 )) )
    cv2.imwrite("img_comp_4_.png", np.uint8( np.fabs(img_dct_4*10.0 )) )
    cv2.imwrite("img_comp_8_.png", np.uint8( np.fabs(img_dct_8*10.0 )) )
    #左上の1/1, 1/2, 1/4, 1/8のみを使って逆離散コサイン変換により複合化したものを出力
    cv2.imwrite("img_comp_1.png", np.uint8( cv2.idct(img_dct_1) ) )
    cv2.imwrite("img_comp_2.png", np.uint8( cv2.idct(img_dct_2) ) )
    cv2.imwrite("img_comp_4.png", np.uint8( cv2.idct(img_dct_4) ) )
    cv2.imwrite("img_comp_8.png", np.uint8( cv2.idct(img_dct_8) ) )
