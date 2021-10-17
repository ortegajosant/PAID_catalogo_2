function resultado = laplaciano(A)
  A=im2double(A);
  B=[1 1 1; 1 -8 1; 1 1 1];
  resultado=conv2(A,B,'same');
  resultado=im2uint8(resultado);
endfunction