clc; clear; close all;
pkg load image

function B=llenado_cuadrantes(A)
  [m,n]=size(A);
  B=zeros(m,n);
  for i=1:round(m/2) 
    for j=1:round(n/2)
      B(i,j)=A(i,j); 
      B(m-i+1,n-j+1)=A(i,j);
      B(m-i+1,j)=A(i,j);
      B(i,n-j+1)=A(i,j);
    endfor
  endfor
endfunction

A=imread('edificio_china.jpg');
subplot(2,2,1)
imshow(A)
title('Imagen Original')


A=im2double(A);

F_A=fftshift(fft2(A));
subplot(2,2,2)
imshow(log(1+real(F_A)), [])
title('DFT2 de A')

[m,n]=size(A);
D0=30; 
k=1;
F_B=zeros(m,n);

for u=1:m
  for v=1:n
    D_uv=sqrt(u**2+v**2);
    F_B(u,v)=1/(1+(D_uv/D0)**(2*k));
  endfor
endfor

F_B=llenado_cuadrantes(F_B);
F_B=fftshift(F_B);

F_C=F_A.*F_B; 
subplot(2,2,3)
imshow(log(1+real(F_C)), [])
title('Resultado de la convolucion')

F_C=fftshift(F_C);

A_t=ifft2(F_C);
A_t=im2uint8(real(A_t));
subplot(2,2,4)
imshow(A_t)
title('DFT2 Inversa con filtro paso bajo')