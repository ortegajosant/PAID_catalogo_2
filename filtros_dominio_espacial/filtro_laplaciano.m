clc;clear;close all
pkg load image

A=imread('baby_yoda.jpg');
subplot(1,2,1)
imshow(A)

B=laplaciano(A);
subplot(1,2,2)
imshow(B)